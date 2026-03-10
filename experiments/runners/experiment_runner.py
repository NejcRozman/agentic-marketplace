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

# Add parent directory to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.runners.metrics_collector import MetricsCollector

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
        self.provider_metadata: List[Dict[str, Any]] = []  # Track provider types and configs
        
        # Experiment directories
        self.experiment_id: str = ""
        self.log_dir: Path = Path()
        self.metrics_dir: Path = Path()
        
        # Status tracking
        self.start_time: Optional[datetime] = None
        self.consumer_status_file: Optional[Path] = None
        
        # Metrics collection
        self.metrics_collector: Optional[MetricsCollector] = None
        self.phase_timings: Dict[str, float] = {}  # Track phase durations
        
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
        
        # Clean existing directories to ensure fresh start
        import shutil
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        if self.metrics_dir.exists():
            shutil.rmtree(self.metrics_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MetricsCollector
        self.metrics_collector = MetricsCollector(
            experiment_id=self.experiment_id,
            metrics_dir=self.metrics_dir,
            logs_dir=self.log_dir
        )
        
        logger.info(f"✓ Configuration loaded for experiment: {self.experiment_id}")
        logger.info(f"  Log directory: {self.log_dir}")
        logger.info(f"  Metrics directory: {self.metrics_dir}")
    
    def cleanup_orphaned_processes(self):
        """Kill any orphaned agent processes from previous interrupted runs."""
        logger.info("Checking for orphaned processes...")
        
        # Kill orphaned consumer processes
        try:
            result = subprocess.run(
                ["pkill", "-f", "consumer_orchestrator.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✓ Killed orphaned consumer processes")
            elif result.returncode == 1:
                logger.debug("No orphaned consumer processes found")
        except Exception as e:
            logger.warning(f"Failed to check for consumer processes: {e}")
        
        # Kill orphaned provider processes
        try:
            result = subprocess.run(
                ["pkill", "-f", "provider_agent/orchestrator.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✓ Killed orphaned provider processes")
            elif result.returncode == 1:
                logger.debug("No orphaned provider processes found")
        except Exception as e:
            logger.warning(f"Failed to check for provider processes: {e}")
        
        # Kill orphaned Anvil processes
        try:
            result = subprocess.run(
                ["pkill", "anvil"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("✓ Killed orphaned Anvil processes")
            elif result.returncode == 1:
                logger.debug("No orphaned Anvil processes found")
        except Exception as e:
            logger.warning(f"Failed to check for Anvil processes: {e}")
    
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
    
    async def fund_consumer_account(self):
        """Fund consumer account with ETH and USDC if using a separate account."""
        logger.info("Checking if consumer needs funding...")
        
        consumer_pk = self.config['blockchain']['accounts']['consumer']
        deployer_pk = self.config['blockchain']['accounts']['deployer']
        
        # Get consumer address
        from eth_account import Account
        consumer_address = Account.from_key(consumer_pk).address
        deployer_address = Account.from_key(deployer_pk).address
        
        # Check if consumer and deployer are different accounts
        if consumer_address.lower() == deployer_address.lower():
            logger.info("Consumer and deployer are the same account - no funding needed")
            return
        
        logger.info(f"Funding consumer account: {consumer_address}")
        
        # Get funding amounts from config
        funding_config = self.config['blockchain']['accounts'].get('consumer_funding', {})
        eth_amount = funding_config.get('eth_amount', 1000000000000000000)  # 1 ETH default
        usdc_amount = funding_config.get('usdc_amount', 1000000000)  # 1000 USDC default
        
        env_vars = {
            "DEPLOYER_PRIVATE_KEY": deployer_pk,
            "CONSUMER_ADDRESS": consumer_address,
            "USDC_ADDRESS": self.payment_token_address,
            "FUND_AMOUNT_ETH": str(eth_amount),
            "FUND_AMOUNT_USDC": str(usdc_amount)
        }
        
        # Run FundConsumer script
        self.run_forge_script("FundConsumer.s.sol", env_vars)
        
        logger.info(f"✓ Consumer account funded")
        logger.info(f"  ETH: {eth_amount / 1e18} ETH")
        logger.info(f"  USDC: {usdc_amount / 1e6} USDC")
    
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
        provider_config = self.config['agents']['providers']

        # Providers for current experiment plan: provider_agent only
        provider_count = provider_config['count']
        self.provider_metadata = []

        for i in range(provider_count):
            pk = provider_pks[i] if i < len(provider_pks) else provider_pks[0]
            agent_id = self.register_agent(pk)
            self.provider_agent_ids.append(agent_id)

            # Store metadata for spawning
            self.provider_metadata.append({
                'agent_id': agent_id,
                'private_key': pk,
                'group_config': provider_config,
                'index': i
            })

            logger.info(f"Provider {i+1} agent ID: {agent_id}")
        
        logger.info(f"✓ All agents registered ({len(self.provider_agent_ids)} providers)")
    
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
            agent_type: "consumer" or "provider_agent"
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
        else:  # provider_agent
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
            if 'provider_pool' in additional_args:
                cmd.extend(["--provider-pool"] + [str(pid) for pid in additional_args['provider_pool']])
            if 'eligible_per_auction' in additional_args:
                cmd.extend(["--eligible-per-auction", str(additional_args['eligible_per_auction'])])
            if 'max_budget' in additional_args:
                cmd.extend(["--max-budget", str(additional_args['max_budget'])])
            if 'auction_duration' in additional_args:
                cmd.extend(["--auction-duration", str(additional_args['auction_duration'])])
            if 'reputation_weight' in additional_args:
                cmd.extend(["--reputation-weight", str(additional_args['reputation_weight'])])
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
        elif agent_type == "provider_agent" and additional_args:
            if 'check_interval' in additional_args:
                cmd.extend(["--check-interval", str(additional_args['check_interval'])])
            if 'architecture' in additional_args:
                cmd.extend(["--architecture", str(additional_args['architecture'])])
            if 'reasoning_mode' in additional_args:
                cmd.extend(["--reasoning-mode", str(additional_args['reasoning_mode'])])
            if 'heuristic_strategy' in additional_args:
                cmd.extend(["--heuristic-strategy", str(additional_args['heuristic_strategy'])])
            if 'heuristic_min_margin' in additional_args:
                cmd.extend(["--heuristic-min-margin", str(additional_args['heuristic_min_margin'])])
            if 'heuristic_max_margin' in additional_args:
                cmd.extend(["--heuristic-max-margin", str(additional_args['heuristic_max_margin'])])
        
        # Setup environment variables
        env = os.environ.copy()
        
        # Add quality profile for providers if specified
        if agent_type == "provider_agent" and additional_args and 'quality_profile' in additional_args:
            env["QUALITY_PROFILE"] = additional_args['quality_profile']
            logger.info(f"Setting QUALITY_PROFILE={additional_args['quality_profile']} for provider {agent_id}")

        # Provider reasoning configuration (also set in env for compatibility)
        if agent_type == "provider_agent" and additional_args:
            if 'architecture' in additional_args:
                env["ARCHITECTURE"] = str(additional_args['architecture'])
            if 'reasoning_mode' in additional_args:
                env["REASONING_MODE"] = str(additional_args['reasoning_mode'])
            if 'heuristic_strategy' in additional_args:
                env["HEURISTIC_STRATEGY"] = str(additional_args['heuristic_strategy'])
            if 'heuristic_min_margin' in additional_args:
                env["HEURISTIC_MIN_MARGIN"] = str(additional_args['heuristic_min_margin'])
            if 'heuristic_max_margin' in additional_args:
                env["HEURISTIC_MAX_MARGIN"] = str(additional_args['heuristic_max_margin'])
            if 'bidding_base_cost' in additional_args:
                env["BIDDING_BASE_COST"] = str(additional_args['bidding_base_cost'])
            if 'gas_cost_mode' in additional_args:
                env["GAS_COST_MODE"] = str(additional_args['gas_cost_mode'])
            if 'gas_price_gwei' in additional_args:
                env["GAS_PRICE_GWEI"] = str(additional_args['gas_price_gwei'])
            if 'eth_price_usd' in additional_args:
                env["ETH_PRICE_USD"] = str(additional_args['eth_price_usd'])
        
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
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
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
            'reputation_weight': consumer_config['config'].get('reputation_weight', 30),
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
            
            # Add random provider selection if configured
            # If eligible_per_auction is set, use registered provider_agent_ids as pool
            if 'eligible_per_auction' in behavior:
                consumer_args['provider_pool'] = self.provider_agent_ids
                consumer_args['eligible_per_auction'] = behavior['eligible_per_auction']
                logger.info(f"Random provider selection enabled: {behavior['eligible_per_auction']} from pool of {len(self.provider_agent_ids)}")
            elif 'provider_pool' in behavior:
                # Fallback: use explicit provider_pool if specified (for backwards compatibility)
                consumer_args['provider_pool'] = behavior['provider_pool']
                if 'eligible_per_auction' in behavior:
                    consumer_args['eligible_per_auction'] = behavior['eligible_per_auction']
        
        self.spawn_agent_process(
            "consumer",
            self.consumer_agent_id,
            consumer_pk,
            consumer_args
        )
        
        # Spawn providers using metadata from registration
        provider_config = self.config['agents']['providers']
        
        for provider_meta in self.provider_metadata:
            agent_id = provider_meta['agent_id']
            pk = provider_meta['private_key']
            group_config = provider_meta['group_config']
            
            # Build provider-specific args
            if 'config' in group_config:
                provider_args = {
                    'check_interval': group_config['config'].get('check_interval', 5)
                }
                
                # Add quality profile if specified (for honest providers)
                if 'quality_profile' in group_config['config']:
                    provider_args['quality_profile'] = group_config['config']['quality_profile']
            else:
                provider_args = {}

            provider_behavior = group_config.get('behavior', {})
            provider_cfg = group_config.get('config', {})

            # Routing for provider strategy selection (YAML-driven)
            architecture = provider_behavior.get('architecture', provider_cfg.get('architecture'))
            reasoning_mode = provider_behavior.get('reasoning_mode', provider_cfg.get('reasoning_mode'))
            heuristic_strategy = provider_behavior.get('heuristic_strategy', provider_cfg.get('heuristic_strategy'))
            heuristic_min_margin = provider_behavior.get('heuristic_min_margin', provider_cfg.get('heuristic_min_margin'))
            heuristic_max_margin = provider_behavior.get('heuristic_max_margin', provider_cfg.get('heuristic_max_margin'))

            if architecture is not None:
                provider_args['architecture'] = architecture
            if reasoning_mode is not None:
                provider_args['reasoning_mode'] = reasoning_mode
            if heuristic_strategy is not None:
                provider_args['heuristic_strategy'] = heuristic_strategy
            if heuristic_min_margin is not None:
                provider_args['heuristic_min_margin'] = heuristic_min_margin
            if heuristic_max_margin is not None:
                provider_args['heuristic_max_margin'] = heuristic_max_margin

            # Provider economic simulation parameters
            bidding_base_cost = provider_cfg.get('bidding_base_cost')
            gas_cost_mode = provider_cfg.get('gas_cost_mode')
            gas_price_gwei = provider_cfg.get('gas_price_gwei')
            eth_price_usd = provider_cfg.get('eth_price_usd')

            if bidding_base_cost is not None:
                provider_args['bidding_base_cost'] = bidding_base_cost
            if gas_cost_mode is not None:
                provider_args['gas_cost_mode'] = gas_cost_mode
            if gas_price_gwei is not None:
                provider_args['gas_price_gwei'] = gas_price_gwei
            if eth_price_usd is not None:
                provider_args['eth_price_usd'] = eth_price_usd

            logger.info(
                f"Provider {agent_id} runtime config: "
                f"architecture={provider_args.get('architecture', 'default')}, "
                f"reasoning_mode={provider_args.get('reasoning_mode', 'default')}, "
                f"heuristic_strategy={provider_args.get('heuristic_strategy', 'default')}"
            )
            
            self.spawn_agent_process(
                "provider_agent",
                agent_id,
                pk,
                provider_args
            )
        
        # Wait for agents to initialize
        logger.info("Waiting for agents to initialize...")
        await asyncio.sleep(3)
        
        logger.info("✓ All agents spawned")
    
    async def snapshot_reputations(self, after_auction_id: int):
        """Capture reputation snapshot after an auction completes.
        
        Args:
            after_auction_id: The auction ID that just completed
        """
        try:
            from web3 import Web3
            from agents.infrastructure.contract_abis import get_reputation_registry_abi
            
            w3 = Web3(Web3.HTTPProvider(self.config['blockchain']['rpc_url']))
            reputation_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.config['blockchain']['contracts']['reputation_registry']),
                abi=get_reputation_registry_abi()
            )
            
            snapshot = {
                "after_auction": after_auction_id,
                "timestamp": datetime.now().isoformat(),
                "reputations": {}
            }
            
            for provider_id in self.provider_agent_ids:
                try:
                    # getSummary returns (feedbackCount, averageScore)
                    feedback_count, average_score = reputation_contract.functions.getSummary(
                        provider_id,
                        [],  # No client address filter
                        bytes(32),  # No tag1 filter
                        bytes(32)   # No tag2 filter
                    ).call()
                    
                    # Default to 50 if no feedback (matches provider agent logic)
                    if feedback_count == 0:
                        average_score = 50
                    
                    snapshot["reputations"][provider_id] = {
                        "score": int(average_score),
                        "feedback_count": int(feedback_count)
                    }
                    logger.info(f"   Provider {provider_id}: score={average_score}, feedback_count={feedback_count}")
                except Exception as e:
                    logger.warning(f"   Failed to get reputation for provider {provider_id}: {e}")
            
            if self.metrics_collector:
                self.metrics_collector.metrics["reputation_evolution"].append(snapshot)
                
        except Exception as e:
            logger.error(f"Failed to snapshot reputations: {e}", exc_info=True)
    
    async def monitor_experiment(self):
        """Monitor experiment progress and check stopping criteria."""
        logger.info("Monitoring experiment progress...")
        
        flow = self.config['experiment_flow']
        stopping_criteria = flow['phases'][2]['stopping_criteria']
        
        criteria_type = stopping_criteria['type']
        # Use consumer's num_auctions as the target (single source of truth)
        # Fall back to stopping_criteria['target'] for backwards compatibility
        target = self.config['agents']['consumer']['behavior'].get('num_auctions', stopping_criteria.get('target', 1))
        poll_interval = stopping_criteria['poll_interval']
    
        max_timeout = self.config['experiment'].get('duration_timeout', stopping_criteria.get('max_timeout', 7200))
        
        logger.info(f"Stopping criteria: {criteria_type}, target={target} auctions, max_timeout={max_timeout}s ({max_timeout/3600:.1f}h)")
        
        # Track reputation snapshots
        reputation_snapshots_enabled = self.config['experiment_flow'].get('reputation_snapshots', {}).get('enabled', True)
        last_completed_count = 0
        
        start_time = time.time()

        # Track consumer process to detect early crashes/exits
        consumer_process = next(
            (p for p in self.agent_processes if p.name.startswith("consumer_")),
            None
        )
        
        while True:
            # If consumer exits unexpectedly, stop waiting immediately
            if consumer_process and consumer_process.process.poll() is not None:
                exit_code = consumer_process.process.returncode
                logger.error(
                    f"Consumer process exited unexpectedly with code {exit_code}. "
                    f"See log: {consumer_process.log_file}"
                )
                break

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                logger.info(f"Maximum timeout reached ({max_timeout}s = {max_timeout/3600:.1f}h)")
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
                        
                        # Take reputation snapshot when a new auction completes
                        if reputation_snapshots_enabled and completed > last_completed_count:
                            logger.info(f"📸 Taking reputation snapshot after auction {completed}...")
                            await self.snapshot_reputations(completed)
                            last_completed_count = completed
                        
                        if completed >= target:
                            logger.info(f"✓ Target reached: {completed} auctions completed")
                            break
                except Exception as e:
                    logger.debug(f"Error reading consumer status: {e}")
            
            # Wait before next check
            await asyncio.sleep(poll_interval)
        
        logger.info("✓ Experiment monitoring complete")
    
    def collect_metrics(self):
        """Collect comprehensive experiment metrics."""
        logger.info("="*80)
        logger.info("Collecting comprehensive experiment metrics...")
        logger.info("="*80)
        
        if not self.metrics_collector:
            logger.error("MetricsCollector not initialized!")
            return
        
        # Set basic info
        self.metrics_collector.set_end_time(datetime.now())
        self.metrics_collector.set_contracts(
            reverse_auction=self.reverse_auction_address,
            payment_token=self.payment_token_address,
            identity_registry=self.config['blockchain']['contracts']['identity_registry'],
            reputation_registry=self.config['blockchain']['contracts']['reputation_registry']
        )
        self.metrics_collector.set_agents(
            consumer_id=self.consumer_agent_id,
            provider_ids=self.provider_agent_ids
        )
        
        # Collect auction data from blockchain
        logger.info("📊 Collecting auction data from blockchain...")
        try:
            from web3 import Web3
            from agents.infrastructure.contract_abis import get_reverse_auction_abi, get_reputation_registry_abi
            
            w3 = Web3(Web3.HTTPProvider(self.config['blockchain']['rpc_url']))
            auction_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.reverse_auction_address),
                abi=get_reverse_auction_abi()
            )
            
            # Get auction count and collect each auction
            auction_count = auction_contract.functions.auctionIdCounter().call()
            logger.info(f"   Found {auction_count} auction(s) on-chain")
            
            summary = self.metrics_collector.metrics["summary"]
            
            # Get reputation contract for querying reputation changes
            reputation_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.config['blockchain']['contracts']['reputation_registry']),
                abi=get_reputation_registry_abi()
            )
            
            for auction_id in range(1, auction_count + 1):  # IDs start at 1, not 0
                logger.info(f"   Collecting auction {auction_id}...")
                auction_data = self.metrics_collector.collect_auction_from_blockchain(
                    w3, auction_contract, auction_id, reputation_contract
                )
                self.metrics_collector.metrics["auctions"].append(auction_data)
                
                # Update summary counts
                summary["total_auctions"] += 1
                if auction_data["status"] == "completed":
                    summary["completed_auctions"] += 1
                elif auction_data["status"] == "failed":
                    summary["failed_auctions"] += 1
                summary["total_bids_attempted"] += len(auction_data["bids_attempted"])
                summary["total_bids_on_chain"] += len(auction_data["bids_on_chain"])
                
                logger.info(f"      Status: {auction_data['status']}")
                logger.info(f"      Bids attempted: {len(auction_data['bids_attempted'])}")
                logger.info(f"      Bids on-chain: {len(auction_data['bids_on_chain'])}")
                logger.info(f"      Winner: {auction_data['winner_id']}")
                logger.info(f"      Quality rating: {auction_data['quality_rating']}/100")
            
            logger.info(f"\n📈 Summary:")
            logger.info(f"   Completed auctions: {summary['completed_auctions']}/{summary['total_auctions']}")
            logger.info(f"   Total bids attempted: {summary['total_bids_attempted']}")
            logger.info(f"   Total bids on-chain: {summary['total_bids_on_chain']}")
            if summary['total_bids_attempted'] > 0:
                success_rate = summary['total_bids_on_chain'] / summary['total_bids_attempted']
                logger.info(f"   Bid success rate: {success_rate:.1%}")
            
        except Exception as e:
            logger.error(f"   Failed to collect auction data: {e}", exc_info=True)
        
        # Collect reputation evolution
        logger.info("\n🏆 Collecting reputation data...")
        try:
            from agents.infrastructure.contract_abis import get_reputation_registry_abi
            
            reputation_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.config['blockchain']['contracts']['reputation_registry']),
                abi=get_reputation_registry_abi()
            )

            final_snapshot = {
                "after_auction": "final",
                "timestamp": datetime.now().isoformat(),
                "source": "final_snapshot",
                "reputations": {}
            }
            
            for provider_id in self.provider_agent_ids:
                try:
                    feedback_count, average_score = reputation_contract.functions.getSummary(
                        provider_id,
                        [],
                        bytes(32),
                        bytes(32)
                    ).call()

                    reputation = int(average_score) if int(feedback_count) > 0 else 50
                    final_snapshot["reputations"][str(provider_id)] = {
                        "score": reputation,
                        "feedback_count": int(feedback_count)
                    }
                    logger.info(f"   Provider {provider_id}: {reputation} (feedback_count={int(feedback_count)})")
                except Exception as e:
                    logger.debug(f"   Could not get reputation for provider {provider_id}: {e}")

            if final_snapshot["reputations"]:
                self.metrics_collector.metrics["reputation_evolution"].append(final_snapshot)
                    
        except Exception as e:
            logger.error(f"   Failed to collect reputation: {e}")
        
        # Collect timing metrics
        logger.info("\n⏱️  Collecting timing metrics...")
        timing = self.metrics_collector.metrics["timing"]
        timing["phase_durations"] = self.phase_timings
        
        total_time = sum(self.phase_timings.values())
        logger.info(f"   Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")
        for phase, duration in self.phase_timings.items():
            logger.info(f"   {phase}: {duration:.1f}s")
        
        # Collect timing data from auction data
        auctions = self.metrics_collector.metrics["auctions"]
        if auctions:
            exec_times = [a["execution_duration_seconds"] for a in auctions if a["execution_duration_seconds"] > 0]
            eval_times = [a["evaluation_duration_seconds"] for a in auctions if a["evaluation_duration_seconds"] > 0]
            
            timing["execution_times"] = exec_times
            timing["evaluation_times"] = eval_times
        
        # Collect errors
        logger.info("\n⚠️  Collecting errors...")
        errors = self.metrics_collector.collect_errors()
        logger.info(f"   Total errors: {errors.get('total_errors')}")
        logger.info(f"   Total warnings: {errors.get('total_warnings')}")
        
        # Highlight critical issues
        if errors.get('critical_issues'):
            logger.warning(f"   ⚠️  {len(errors['critical_issues'])} critical issue(s) detected")
        
        # Determine overall success
        success = len(auctions) > 0 and any(a["status"] == "completed" for a in auctions)
        self.metrics_collector.metrics["success"] = success
        logger.info(f"\n{'✅' if success else '❌'} Experiment success: {success}")
        
        # Save metrics
        metrics_file = self.metrics_collector.save_metrics()
        logger.info("="*80)
        logger.info(f"📁 Comprehensive metrics saved to: {metrics_file}")
        logger.info("="*80)
    
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
            # Load configuration (initializes metrics_collector)
            self.load_config()
            
            # Kill any orphaned processes from previous runs
            self.cleanup_orphaned_processes()
            
            # Set start time after metrics_collector is initialized
            if self.metrics_collector:
                self.metrics_collector.set_start_time(self.start_time)
            
            # Phase 1: Setup blockchain
            logger.info("=== Phase 1: Setup Blockchain ===")
            phase_start = datetime.now()
            await self.start_anvil()
            await self.deploy_contracts()
            await self.fund_consumer_account()
            await self.register_agents()
            self.phase_timings["setup_blockchain"] = (datetime.now() - phase_start).total_seconds()
            logger.info(f"✓ Phase 1 completed in {self.phase_timings['setup_blockchain']:.1f}s")
            
            # Phase 2: Start agents
            logger.info("=== Phase 2: Start Agents ===")
            phase_start = datetime.now()
            await self.spawn_agents()
            self.phase_timings["start_agents"] = (datetime.now() - phase_start).total_seconds()
            logger.info(f"✓ Phase 2 completed in {self.phase_timings['start_agents']:.1f}s")
            
            # Phase 3: Run experiment
            logger.info("=== Phase 3: Run Experiment ===")
            phase_start = datetime.now()
            await self.monitor_experiment()
            self.phase_timings["experiment_run"] = (datetime.now() - phase_start).total_seconds()
            logger.info(f"✓ Phase 3 completed in {self.phase_timings['experiment_run']:.1f}s")
            
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