"""Blockchain client for interacting with smart contracts."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from web3 import Web3, AsyncWeb3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_utils import to_checksum_address

from .config import config, BlockchainConfig

logger = logging.getLogger(__name__)


class BlockchainClient:
    """
    Asynchronous blockchain client for interacting with Ethereum-compatible networks.
    
    Provides methods for:
    - Contract interaction
    - Transaction management
    - Event monitoring
    - Account management
    """
    
    def __init__(self, blockchain_config: Optional[BlockchainConfig] = None):
        self.config = blockchain_config or config.blockchain
        self.w3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.contracts: Dict[str, Any] = {}
        
        # Initialize connection
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the blockchain connection."""
        try:
            # Create Web3 instance
            self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.config.rpc_url))
            
            # Add PoA middleware if needed (for networks like BSC, Polygon)
            if self.config.chain_id in [56, 137, 80001]:  # BSC, Polygon, Mumbai
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Set up account if private key is provided
            if self.config.private_key:
                self.account = Account.from_key(self.config.private_key)
                logger.info(f"Initialized account: {self.account.address}")
            
            # Verify connection
            if await self.is_connected():
                chain_id = await self.w3.eth.chain_id
                logger.info(f"Connected to blockchain - Chain ID: {chain_id}")
            else:
                logger.error("Failed to connect to blockchain")
                
        except Exception as e:
            logger.error(f"Error initializing blockchain client: {e}")
            raise
    
    async def is_connected(self) -> bool:
        """Check if connected to the blockchain."""
        try:
            if not self.w3:
                return False
            await self.w3.eth.block_number
            return True
        except Exception:
            return False
    
    async def get_balance(self, address: Optional[str] = None) -> int:
        """Get ETH/native token balance for an address."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        addr = address or (self.account.address if self.account else None)
        if not addr:
            raise ValueError("No address provided and no account configured")
        
        balance = await self.w3.eth.get_balance(to_checksum_address(addr))
        return balance
    
    async def get_gas_price(self) -> int:
        """Get current gas price."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        if self.config.gas_price:
            return self.config.gas_price
        
        return await self.w3.eth.gas_price
    
    async def load_contract(self, name: str, address: str, abi: List[Dict]) -> Any:
        """Load a smart contract."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.w3.eth.contract(
            address=to_checksum_address(address),
            abi=abi
        )
        
        self.contracts[name] = contract
        logger.info(f"Loaded contract {name} at {address}")
        return contract
    
    async def call_contract_method(
        self,
        contract_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Call a read-only contract method."""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)
        
        return await method(*args, **kwargs).call()
    
    async def send_transaction(
        self,
        contract_name: str,
        method_name: str,
        *args,
        value: int = 0,
        gas_limit: Optional[int] = None,
        gas_price: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send a transaction to a contract method."""
        if not self.account:
            raise RuntimeError("No account configured for sending transactions")
        
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)
        
        # Build transaction
        transaction = await method(*args, **kwargs).build_transaction({
            'from': self.account.address,
            'value': value,
            'gas': gas_limit or self.config.gas_limit,
            'gasPrice': gas_price or await self.get_gas_price(),
            'nonce': await self.w3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id,
        })
        
        # Sign transaction
        signed_txn = self.account.sign_transaction(transaction)
        
        # Send transaction
        tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        logger.info(f"Sent transaction: {tx_hash.hex()}")
        return tx_hash.hex()
    
    async def wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for transaction confirmation."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        try:
            receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            
            return {
                'transactionHash': receipt['transactionHash'].hex(),
                'blockNumber': receipt['blockNumber'],
                'gasUsed': receipt['gasUsed'],
                'status': receipt['status'],
                'logs': [dict(log) for log in receipt['logs']]
            }
        except Exception as e:
            logger.error(f"Error waiting for transaction {tx_hash}: {e}")
            raise
    
    async def get_contract_events(
        self,
        contract_name: str,
        event_name: str,
        from_block: Union[int, str] = 'latest',
        to_block: Union[int, str] = 'latest',
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Get contract events."""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        contract = self.contracts[contract_name]
        event = getattr(contract.events, event_name)
        
        event_filter = event.create_filter(
            fromBlock=from_block,
            toBlock=to_block,
            argument_filters=filters or {}
        )
        
        events = await event_filter.get_all_entries()
        return [dict(event) for event in events]
    
    async def estimate_gas(
        self,
        contract_name: str,
        method_name: str,
        *args,
        value: int = 0,
        **kwargs
    ) -> int:
        """Estimate gas for a transaction."""
        if not self.account:
            raise RuntimeError("No account configured")
        
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not loaded")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        contract = self.contracts[contract_name]
        method = getattr(contract.functions, method_name)
        
        gas_estimate = await method(*args, **kwargs).estimate_gas({
            'from': self.account.address,
            'value': value
        })
        
        return gas_estimate
    
    async def deploy_contract(
        self,
        bytecode: str,
        abi: List[Dict],
        constructor_args: Optional[List] = None,
        gas_limit: Optional[int] = None
    ) -> str:
        """Deploy a new contract."""
        if not self.account:
            raise RuntimeError("No account configured for deployment")
        
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        # Create contract instance
        contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Build deployment transaction
        transaction = await contract.constructor(
            *(constructor_args or [])
        ).build_transaction({
            'from': self.account.address,
            'gas': gas_limit or self.config.gas_limit,
            'gasPrice': await self.get_gas_price(),
            'nonce': await self.w3.eth.get_transaction_count(self.account.address),
            'chainId': self.config.chain_id,
        })
        
        # Sign and send transaction
        signed_txn = self.account.sign_transaction(transaction)
        tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for deployment
        receipt = await self.wait_for_transaction(tx_hash.hex())
        
        if receipt['status'] == 1:
            logger.info(f"Contract deployed at: {receipt.get('contractAddress')}")
            return receipt.get('contractAddress')
        else:
            raise RuntimeError(f"Contract deployment failed: {receipt}")
    
    async def get_block(self, block_identifier: Union[int, str] = 'latest') -> Dict[str, Any]:
        """Get block information."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        block = await self.w3.eth.get_block(block_identifier)
        return dict(block)
    
    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction information."""
        if not self.w3:
            raise RuntimeError("Blockchain client not initialized")
        
        tx = await self.w3.eth.get_transaction(tx_hash)
        return dict(tx)