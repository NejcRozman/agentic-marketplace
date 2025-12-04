// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {ReverseAuction} from "../src/ReverseAuction.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";

/**
 * @title Deploy Script
 * @notice Deploys ReverseAuction contract with mock USDC to a forked Sepolia network
 * 
 * Usage:
 *   1. Start Anvil with Sepolia fork:
 *      anvil --fork-url https://sepolia.infura.io/v3/${INFURA_API_KEY}
 * 
 *   2. Run this script:
 *      forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
 * 
 * The script will:
 *   - Deploy a mock USDC token
 *   - Deploy ReverseAuction with the ERC-8004 registries from Sepolia
 *   - Mint USDC to test accounts for auction creation
 */
contract DeployScript is Script {
    // Anvil's pre-funded test accounts (public keys - safe to hardcode)
    address constant DEPLOYER = 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266;
    address constant BUYER = 0x70997970C51812dc3A010C7d01b50e0d17dc79C8;
    address constant PROVIDER = 0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC;
    
    // Anvil's private keys (these are public test keys)
    uint256 constant DEPLOYER_PK = 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;
    
    function run() external {
        // Load ERC-8004 addresses from environment
        address identityRegistry = vm.envAddress("IDENTITY_REGISTRY_ADDRESS");
        address reputationRegistry = vm.envAddress("REPUTATION_REGISTRY_ADDRESS");
        
        console.log("=== Deploy Configuration ===");
        console.log("Deployer:", DEPLOYER);
        console.log("Buyer:", BUYER);
        console.log("Provider:", PROVIDER);
        console.log("IdentityRegistry:", identityRegistry);
        console.log("ReputationRegistry:", reputationRegistry);
        console.log("");
        
        vm.startBroadcast(DEPLOYER_PK);
        
        // 1. Deploy Mock USDC
        ERC20Mock usdc = new ERC20Mock();
        console.log("Mock USDC deployed at:", address(usdc));
        
        // 2. Deploy ReverseAuction
        ReverseAuction auction = new ReverseAuction(
            address(usdc),
            identityRegistry,
            reputationRegistry
        );
        console.log("ReverseAuction deployed at:", address(auction));
        
        // 3. Mint USDC to test accounts
        uint256 mintAmount = 100_000 * 1e6; // 100,000 USDC (6 decimals like real USDC)
        usdc.mint(BUYER, mintAmount);
        usdc.mint(PROVIDER, mintAmount);
        console.log("Minted", mintAmount / 1e6, "USDC to Buyer and Provider");
        
        vm.stopBroadcast();
        
        console.log("");
        console.log("=== Deployment Complete ===");
        console.log("Add these to your agents/.env:");
        console.log("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS=", address(auction));
        console.log("MOCK_USDC_ADDRESS=", address(usdc));
    }
}
