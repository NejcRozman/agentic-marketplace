// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {IIdentityRegistry} from "../src/interfaces/IIdentityRegistry.sol";
import {AgentProxy} from "./AgentProxy.sol";

/**
 * @title RegisterAgent Script
 * @notice Deploys an AgentProxy and registers it on ERC-8004 IdentityRegistry
 * 
 * Why AgentProxy?
 *   The ERC-8004 IdentityRegistry uses _safeMint internally, which calls onERC721Received
 *   on the recipient. EOAs (like Anvil's pre-funded accounts) cannot implement this interface,
 *   so we deploy a proxy contract that can receive NFTs and act on behalf of the EOA.
 * 
 * Usage:
 *   1. Make sure Anvil is running with Sepolia fork
 *   2. Run this script:
 *      forge script script/RegisterAgent.s.sol --rpc-url localhost --broadcast
 * 
 * The script will:
 *   - Deploy an AgentProxy contract owned by the Provider account
 *   - Register the proxy as an agent on the IdentityRegistry
 *   - Output the agent ID and proxy address to use in tests
 */
contract RegisterAgentScript is Script {
    // Anvil's pre-funded test accounts
    address constant PROVIDER = 0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC;
    
    // Anvil's private key for Provider (Account 2)
    uint256 constant PROVIDER_PK = 0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a;
    
    function run() external {
        // Load IdentityRegistry address from environment
        address identityRegistryAddr = vm.envAddress("IDENTITY_REGISTRY_ADDRESS");
        
        console.log("=== Register Agent via Proxy ===");
        console.log("Provider (owner):", PROVIDER);
        console.log("IdentityRegistry:", identityRegistryAddr);
        console.log("");
        
        vm.startBroadcast(PROVIDER_PK);
        
        // Deploy AgentProxy
        AgentProxy proxy = new AgentProxy(identityRegistryAddr);
        console.log("AgentProxy deployed at:", address(proxy));
        
        // Register the proxy as an agent
        uint256 agentId = proxy.register();
        
        vm.stopBroadcast();
        
        // Verify registration
        IIdentityRegistry identityRegistry = IIdentityRegistry(identityRegistryAddr);
        address agentOwner = identityRegistry.ownerOf(agentId);
        
        console.log("");
        console.log("=== Registration Successful ===");
        console.log("Agent ID:", agentId);
        console.log("Agent (proxy) address:", address(proxy));
        console.log("Agent owner (NFT holder):", agentOwner);
        console.log("");
        console.log("Add these to your test configuration:");
        console.log("  AGENT_ID=", agentId);
        console.log("  AGENT_PROXY_ADDRESS=", address(proxy));
    }
}
