"""
Experiment Runner - Orchestrates multi-agent simulation experiments.

Responsibilities:
1. Load YAML experiment configuration
2. Start Anvil blockchain with Sepolia fork
3. Deploy contracts (ReverseAuction, Mock USDC)
4. Register agents (consumer + providers)
5. Spawn agent processes with appropriate configuration
6. Monitor experiment progress and stopping criteria
7. Collect metrics and logs
8. Clean shutdown of all processes
"""

import asyncio
import subprocess
import logging
import yaml
import json
import os
import sys
import re
import signal
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from agents/.env
AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"
ENV_FILE = AGENTS_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

logger = logging.getLogger(__name__)


@dataclass
class ProcessHandle:
    """Handle for a managed subprocess."""
    name: str
    process: subprocess.Popen
    log_file: Path
    pid: int


class ExperimentRunner:
    """Orchestrates execution of simulation experiments."""
    
    def __init__(self, config_path: Path):
        """Initialize experiment runner with configuration."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.processes: List[ProcessHandle] = []
        
        # Runtime state
        self.anvil_process: Optional[ProcessHandle] = None
        self.agent_processes: List[ProcessHandle] = []
        
        # Deployed contract addresses
        self.reverse_auction_address: Optional[str] = None
        self.payment_token_address: Optional[str] = None
        
        # Registered agent IDs
        self.consumer_agent_id: Optional[int] = None
        self.provider_agent_ids: List[int] = []
        
        # Experiment directories
        self.experiment_id: str = ""
        self.log_dir: Path = Path()
        self.metrics_dir: Path = Path()
        
        # Status tracking
        self.start_time: Optional[datetime] = None
        self.consumer_status_file: Optional[Path] = None
        
    def load_config(self):
        """Load and validate experiment configuration from YAML."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_text = f.read()
        
        # Substitute environment variables (${VAR_NAME} pattern)
        def replace_env_var(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable {var_name} not found")
            return value
        
        config_text = re.sub(r'\$\{(\w+)\}', replace_env_var, config_text)
        
        # Parse YAML
        self.config = yaml.safe_load(config_text)
        
        # Extract key configuration
        self.experiment_id = self.config['experiment']['id']
        
        # Setup experiment directories
        base_dir = Path(__file__).parent.parent / "data"
        self.log_dir = base_dir / "logs" / self.experiment_id
        self.metrics_dir = base_dir / "metrics" / self.experiment_id
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Configuration loaded for experiment: {self.experiment_id}")
        logger.info(f"  Log directory: {self.log_dir}")
        logger.info(f"  Metrics directory: {self.metrics_dir}")
    
    async def start_anvil(self):
        """Start Anvil blockchain with Sepolia fork."""
        logger.info("Starting Anvil blockchain...")
        
        fork_network = self.config['blockchain']['fork_network']
        infura_key_env = self.config['blockchain']['infura_key_env']
        infura_key = os.getenv(infura_key_env)
        
        if not infura_key:
            raise ValueError(f"Environment variable {infura_key_env} not found")
        
        fork_url = f"https://{fork_network}.infura.io/v3/{infura_key}"
        
        # Start Anvil with minimal accounts to reduce Infura requests
        log_file = self.log_dir / "anvil.log"
        
        # Use only one funded account - Anvil's first default key but with massive balance
        # This avoids Anvil trying to fetch state for 10 default accounts from Sepolia
        anvil_default_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                [
                    "anvil", 
                    "--fork-url", fork_url, 
                    "--host", "0.0.0.0",
                    "--accounts", "1",  # Only create 1 default account to minimize Infura requests
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
        
        self.anvil_process = ProcessHandle(
            name="anvil",
            process=process,
            log_file=log_file,
            pid=process.pid
        )
        self.processes.append(self.anvil_process)
        
        # Wait for Anvil to be ready - check if RPC is accepting connections
        logger.info("Waiting for Anvil to initialize...")
        max_wait = 30  # 30 seconds max
        for i in range(max_wait):
            await asyncio.sleep(1)
            
            # Check if Anvil process is still running
            if self.anvil_process.process.poll() is not None:
                raise RuntimeError("Anvil process terminated unexpectedly")
            
            # Try to connect to RPC
            try:
                result = subprocess.run(
                    ["curl", "-s", "-X", "POST", 
                     "-H", "Content-Type: application/json",
                     "--data", '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
                     self.config['blockchain']['rpc_url']],
                    capture_output=True,
                    timeout=1
                )
                if result.returncode == 0 and b"result" in result.stdout:
                    logger.info(f"✓ Anvil ready after {i+1} seconds")
                    break
            except:
                pass
        else:
            raise RuntimeError("Anvil failed to start within 30 seconds")
        
        logger.info(f"✓ Anvil started (PID: {self.anvil_process.pid})")
        
        # Reset nonce for the personal account to 0
        # This is needed because forked accounts carry over their nonce from Sepolia
        logger.info("Resetting account nonce...")
        personal_account = self.config['blockchain']['accounts']['main_account']
        # Get address from private key
        from eth_account import Account
        account = Account.from_key(personal_account)
        address = account.address
        
        # Reset nonce to 0 using anvil_setNonce
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "-H", "Content-Type: application/json",
             "--data", f'{{"jsonrpc":"2.0","method":"anvil_setNonce","params":["{address}","0x0"],"id":1}}',
             self.config['blockchain']['rpc_url']],
            capture_output=True
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Reset nonce for {address}")
        else:
            logger.warning(f"Failed to reset nonce: {result.stderr.decode()}")
    
    def run_forge_script(self, script_name: str, env_vars: Dict[str, str]) -> str:
        """
        Run a forge script and return stdout.
        
        Args:
            script_name: Name of the script file (e.g., "Deploy.s.sol")
            env_vars: Additional environment variables to set
        
        Returns:
            stdout from the script execution
        """
        logger.info(f"Running forge script: {script_name}")
        
        contracts_dir = Path(__file__).parent.parent.parent / "contracts"
        script_path = contracts_dir / "script" / script_name
        
        # Prepare environment
        env = os.environ.copy()
        env.update(env_vars)
        
        # Run forge script
        result = subprocess.run(
            [
                "forge", "script", str(script_path),
                "--rpc-url", self.config['blockchain']['rpc_url'],
                "--broadcast"
            ],
            cwd=contracts_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Forge script failed:\n{result.stderr}")
            raise RuntimeError(f"Forge script {script_name} failed")
        
        logger.info(f"✓ {script_name} executed successfully")
        return result.stdout
    
    async def deploy_contracts(self):
        """Deploy ReverseAuction and Mock USDC contracts."""
        logger.info("Deploying contracts...")
        
        # Get deployer private key (using main account from config)
        private_key = self.config['blockchain']['accounts']['deployer']
        
        env_vars = {
            "PRIVATE_KEY": private_key,
            "IDENTITY_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['identity_registry'],
            "REPUTATION_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['reputation_registry']
        }
        
        # Run Deploy script
        output = self.run_forge_script("Deploy.s.sol", env_vars)
        
        # Parse deployed addresses from output
        # Looking for: "Mock USDC deployed at: 0x..."
        # and "ReverseAuction deployed at: 0x..."
        usdc_match = re.search(r'Mock USDC deployed at:\s*(0x[a-fA-F0-9]+)', output)
        auction_match = re.search(r'ReverseAuction deployed at:\s*(0x[a-fA-F0-9]+)', output)
        
        if not usdc_match or not auction_match:
            raise RuntimeError("Failed to parse contract addresses from deploy script output")
        
        self.payment_token_address = usdc_match.group(1)
        self.reverse_auction_address = auction_match.group(1)
        
        logger.info(f"✓ Contracts deployed:")
        logger.info(f"  Mock USDC: {self.payment_token_address}")
        logger.info(f"  ReverseAuction: {self.reverse_auction_address}")
    
    def register_agent(self, private_key: str) -> int:
        """
        Register an agent and return the agent ID.
        
        Args:
            private_key: Private key to use for registration
        
        Returns:
            Registered agent ID
        """
        env_vars = {
            "PRIVATE_KEY": private_key,
            "IDENTITY_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['identity_registry']
        }
        
        # Run RegisterAgent script
        output = self.run_forge_script("RegisterAgent.s.sol", env_vars)
        
        # Parse agent ID from output
        # Looking for: "Agent ID: 1234"
        match = re.search(r'Agent ID:\s*(\d+)', output)
        
        if not match:
            raise RuntimeError("Failed to parse agent ID from registration output")
        
        agent_id = int(match.group(1))
        logger.info(f"✓ Agent registered with ID: {agent_id}")
        
        return agent_id
    
    async def register_agents(self):
        """Register consumer and provider agents."""
        logger.info("Registering agents...")
        
        # Register consumer
        consumer_pk = self.config['blockchain']['accounts']['consumer']
        self.consumer_agent_id = self.register_agent(consumer_pk)
        logger.info(f"Consumer agent ID: {self.consumer_agent_id}")
        
        # Register providers
        provider_pks = self.config['blockchain']['accounts']['providers']
        provider_count = self.config['agents']['providers']['count']
        
        for i in range(provider_count):
            pk = provider_pks[i] if i < len(provider_pks) else provider_pks[0]
            agent_id = self.register_agent(pk)
            self.provider_agent_ids.append(agent_id)
            logger.info(f"Provider {i+1} agent ID: {agent_id}")
        
        logger.info(f"✓ All agents registered")
    
    def spawn_agent_process(
        self,
        agent_type: str,
        agent_id: int,
        private_key: str,
        additional_args: Dict[str, Any] = None
    ) -> ProcessHandle:
        """
        Spawn an agent process (consumer or provider).
        
        Args:
            agent_type: "consumer" or "provider"
            agent_id: Agent ID to use
            private_key: Blockchain private key
            additional_args: Additional CLI arguments
        
        Returns:
            ProcessHandle for the spawned process
        """
        logger.info(f"Spawning {agent_type} agent (ID: {agent_id})...")
        
        # Determine script path
        agents_dir = Path(__file__).parent.parent.parent / "agents"
        
        if agent_type == "consumer":
            script = agents_dir / "consumer_agent" / "consumer_orchestrator.py"
            status_file = self.log_dir / f"consumer_{agent_id}_status.json"
            self.consumer_status_file = status_file
        else:
            script = agents_dir / "provider_agent" / "orchestrator.py"
            status_file = self.log_dir / f"provider_{agent_id}_status.json"
        
        # Build command - use same Python interpreter as the runner (preserves venv)
        cmd = [
            sys.executable, str(script),
            "--agent-id", str(agent_id),
            "--private-key", private_key,
            "--status-file", str(status_file),
            "--log-level", "INFO"
        ]
        
        # Add type-specific arguments
        if agent_type == "consumer" and additional_args:
            if 'eligible_providers' in additional_args:
                cmd.extend(["--eligible-providers"] + [str(pid) for pid in additional_args['eligible_providers']])
            if 'max_budget' in additional_args:
                cmd.extend(["--max-budget", str(additional_args['max_budget'])])
            if 'auction_duration' in additional_args:
                cmd.extend(["--auction-duration", str(additional_args['auction_duration'])])
            if 'check_interval' in additional_args:
                cmd.extend(["--check-interval", str(additional_args['check_interval'])])
            # Auto-auction arguments
            if 'auto_create_auction' in additional_args and additional_args['auto_create_auction']:
                cmd.append("--auto-create-auction")
            if 'num_auctions' in additional_args:
                cmd.extend(["--num-auctions", str(additional_args['num_auctions'])])
            if 'pdf_directory' in additional_args:
                cmd.extend(["--pdf-directory", str(additional_args['pdf_directory'])])
            if 'service_complexity' in additional_args:
                cmd.extend(["--service-complexity", str(additional_args['service_complexity'])])
            if 'auction_creation_delay' in additional_args:
                cmd.extend(["--auction-creation-delay", str(additional_args['auction_creation_delay'])])
            if 'inter_auction_delay' in additional_args:
                cmd.extend(["--inter-auction-delay", str(additional_args['inter_auction_delay'])])
        elif agent_type == "provider" and additional_args:
            if 'check_interval' in additional_args:
                cmd.extend(["--check-interval", str(additional_args['check_interval'])])
        
        # Setup environment variables
        env = os.environ.copy()
        
        # Add project root to PYTHONPATH so agents can import from agents package
        project_root = str(Path(__file__).parent.parent.parent)
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = project_root
        
        env.update({
            "BLOCKCHAIN_RPC_URL": self.config['blockchain']['rpc_url'],
            "BLOCKCHAIN_REVERSE_AUCTION_ADDRESS": self.reverse_auction_address,
            "BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['identity_registry'],
            "BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS": self.config['blockchain']['contracts']['reputation_registry'],
            "BLOCKCHAIN_PAYMENT_TOKEN_ADDRESS": self.payment_token_address,
            "BLOCKCHAIN_PRIVATE_KEY": private_key,
            "BLOCKCHAIN_AGENT_ID": str(agent_id),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "PINATA_API_KEY": os.getenv("PINATA_API_KEY", ""),
            "PINATA_API_SECRET": os.getenv("PINATA_API_SECRET", ""),
        })
        
        # Start process
        log_file = self.log_dir / f"{agent_type}_{agent_id}.log"
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        handle = ProcessHandle(
            name=f"{agent_type}_{agent_id}",
            process=process,
            log_file=log_file,
            pid=process.pid
        )
        
        self.agent_processes.append(handle)
        self.processes.append(handle)
        
        logger.info(f"✓ {agent_type.capitalize()} agent spawned (PID: {handle.pid})")
        return handle
    
    async def spawn_agents(self):
        """Spawn all consumer and provider agent processes."""
        logger.info("Spawning agent processes...")
        
        # Spawn consumer
        consumer_config = self.config['agents']['consumer']
        consumer_pk = self.config['blockchain']['accounts']['consumer']
        
        consumer_args = {
            'eligible_providers': self.provider_agent_ids,
            'max_budget': consumer_config['config']['max_budget'],
            'auction_duration': consumer_config['config']['auction_duration'],
            'check_interval': consumer_config['config']['check_interval']
        }
        
        # Add behavior config for auto-auction
        if 'behavior' in consumer_config:
            behavior = consumer_config['behavior']
            consumer_args.update({
                'auto_create_auction': behavior.get('auto_create_auction', False),
                'num_auctions': behavior.get('num_auctions', 1),
                'pdf_directory': behavior.get('pdf_directory', ''),
                'service_complexity': behavior.get('service_complexity', 'medium'),
                'auction_creation_delay': behavior.get('auction_creation_delay', 0),
                'inter_auction_delay': behavior.get('inter_auction_delay', 30)
            })
        
        self.spawn_agent_process(
            "consumer",
            self.consumer_agent_id,
            consumer_pk,
            consumer_args
        )
        
        # Spawn providers
        provider_config = self.config['agents']['providers']
        provider_pks = self.config['blockchain']['accounts']['providers']
        
        provider_args = {
            'check_interval': provider_config['config']['check_interval']
        }
        
        for i, agent_id in enumerate(self.provider_agent_ids):
            pk = provider_pks[i] if i < len(provider_pks) else provider_pks[0]
            self.spawn_agent_process(
                "provider",
                agent_id,
                pk,
                provider_args
            )
        
        # Wait for agents to initialize
        logger.info("Waiting for agents to initialize...")
        await asyncio.sleep(10)
        
        logger.info("✓ All agents spawned")
    
    async def monitor_experiment(self):
        """Monitor experiment progress and check stopping criteria."""
        logger.info("Monitoring experiment progress...")
        
        flow = self.config['experiment_flow']
        stopping_criteria = flow['phases'][2]['stopping_criteria']
        
        criteria_type = stopping_criteria['type']
        target = stopping_criteria['target']
        poll_interval = stopping_criteria['poll_interval']
        max_timeout = stopping_criteria['max_timeout']
        
        start_time = time.time()
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                logger.info(f"Maximum timeout reached ({max_timeout}s)")
                break
            
            # Check stopping criteria
            if criteria_type == "completed_auctions":
                try:
                    # Read consumer status file
                    if self.consumer_status_file and self.consumer_status_file.exists():
                        with open(self.consumer_status_file, 'r') as f:
                            status = json.load(f)
                        
                        completed = status.get('completed_auctions', 0)
                        logger.info(f"Completed auctions: {completed}/{target} (elapsed: {int(elapsed)}s)")
                        
                        if completed >= target:
                            logger.info(f"✓ Target reached: {completed} auctions completed")
                            break
                except Exception as e:
                    logger.debug(f"Error reading consumer status: {e}")
            
            # Wait before next check
            await asyncio.sleep(poll_interval)
        
        logger.info("✓ Experiment monitoring complete")
    
    def collect_metrics(self):
        """Collect final experiment metrics."""
        logger.info("Collecting experiment metrics...")
        
        metrics = {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": datetime.now().isoformat(),
            "contracts": {
                "reverse_auction": self.reverse_auction_address,
                "payment_token": self.payment_token_address
            },
            "agents": {
                "consumer_id": self.consumer_agent_id,
                "provider_ids": self.provider_agent_ids
            }
        }
        
        # Read final consumer status
        if self.consumer_status_file and self.consumer_status_file.exists():
            try:
                with open(self.consumer_status_file, 'r') as f:
                    metrics['consumer_final_status'] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read consumer final status: {e}")
        
        # Write metrics
        metrics_file = self.metrics_dir / "experiment_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Metrics saved to {metrics_file}")
    
    def cleanup(self):
        """Shutdown all processes gracefully."""
        logger.info("Shutting down processes...")
        
        # Shutdown agents first (graceful)
        for handle in self.agent_processes:
            try:
                logger.info(f"Stopping {handle.name} (PID: {handle.pid})...")
                os.killpg(os.getpgid(handle.pid), signal.SIGTERM)
                handle.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping {handle.name}: {e}")
                try:
                    os.killpg(os.getpgid(handle.pid), signal.SIGKILL)
                except:
                    pass
        
        # Shutdown Anvil
        if self.anvil_process:
            try:
                logger.info(f"Stopping Anvil (PID: {self.anvil_process.pid})...")
                os.killpg(os.getpgid(self.anvil_process.pid), signal.SIGTERM)
                self.anvil_process.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping Anvil: {e}")
                try:
                    os.killpg(os.getpgid(self.anvil_process.pid), signal.SIGKILL)
                except:
                    pass
        
        logger.info("✓ All processes stopped")
    
    async def run(self):
        """Run the complete experiment."""
        self.start_time = datetime.now()
        
        try:
            # Load configuration
            self.load_config()
            
            # Phase 1: Setup blockchain
            logger.info("=== Phase 1: Setup Blockchain ===")
            await self.start_anvil()
            await self.deploy_contracts()
            await self.register_agents()
            
            # Phase 2: Start agents
            logger.info("=== Phase 2: Start Agents ===")
            await self.spawn_agents()
            
            # Phase 3: Run experiment
            logger.info("=== Phase 3: Run Experiment ===")
            await self.monitor_experiment()
            
            # Collect metrics
            self.collect_metrics()
            
            logger.info("=== Experiment Complete ===")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run experiment simulation")
    parser.add_argument("config", type=Path, help="Path to experiment YAML configuration")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiment
    runner = ExperimentRunner(args.config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
