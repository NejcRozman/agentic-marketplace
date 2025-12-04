// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2 as console} from "forge-std/Script.sol";
import {ReverseAuction} from "../src/ReverseAuction.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title CreateAuction Script
 * @notice Creates a test auction for the provider agent to discover and bid on
 * 
 * Usage:
 *   1. Make sure Anvil is running with Sepolia fork
 *   2. Make sure Deploy.s.sol has been run
 *   3. Run this script:
 *      forge script script/CreateAuction.s.sol --rpc-url localhost --broadcast
 * 
 * The script will:
 *   - Approve USDC spending for the ReverseAuction contract
 *   - Create a new auction with test parameters
 *   - Output the auction ID for testing
 */
contract CreateAuctionScript is Script {
    // Anvil's pre-funded test accounts
    address constant BUYER = 0x70997970C51812dc3A010C7d01b50e0d17dc79C8;
    
    // Anvil's private key for Buyer (Account 1)
    uint256 constant BUYER_PK = 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d;
    
    function run() external {
        // Load contract addresses from environment
        address reverseAuctionAddr = vm.envAddress("REVERSE_AUCTION_ADDRESS");
        address usdcAddr = vm.envAddress("MOCK_USDC_ADDRESS");
        
        console.log("=== Create Auction ===");
        console.log("Buyer:", BUYER);
        console.log("ReverseAuction:", reverseAuctionAddr);
        console.log("USDC:", usdcAddr);
        console.log("Eligible Agent ID:", vm.envUint("AGENT_ID"));
        console.log("");
        
        ReverseAuction reverseAuction = ReverseAuction(reverseAuctionAddr);
        IERC20 usdc = IERC20(usdcAddr);
        
        // Load the provider's agent ID
        uint256 agentId = vm.envUint("AGENT_ID");
        
        // Load service description CID from environment (or use default for testing)
        string memory serviceDescriptionCID = vm.envOr(
            "SERVICE_DESCRIPTION_CID",
            string("QmTestServiceDescriptionCID123456789")
        );
        
        // Auction parameters
        uint256 maxBudget = 100 * 1e6; // 100 USDC (6 decimals)
        uint256 duration = 1 hours;
        uint256 reputationWeight = 50; // 50% reputation, 50% price
        
        // Create eligible agents array with just our provider's agent
        uint256[] memory eligibleAgentIds = new uint256[](1);
        eligibleAgentIds[0] = agentId;
        
        vm.startBroadcast(BUYER_PK);
        
        // Check buyer's USDC balance
        uint256 balance = usdc.balanceOf(BUYER);
        console.log("Buyer USDC balance:", balance / 1e6, "USDC");
        require(balance >= maxBudget, "Insufficient USDC balance");
        
        // Approve ReverseAuction to spend USDC
        usdc.approve(reverseAuctionAddr, maxBudget);
        console.log("Approved ReverseAuction to spend", maxBudget / 1e6, "USDC");
        
        // Create the auction
        uint256 auctionId = reverseAuction.createAuction(
            serviceDescriptionCID,
            maxBudget,
            duration,
            eligibleAgentIds,
            reputationWeight
        );
        
        vm.stopBroadcast();
        
        // Get auction details
        ReverseAuction.Auction memory auction = reverseAuction.getAuctionDetails(auctionId);
        
        console.log("");
        console.log("=== Auction Created ===");
        console.log("Auction ID:", auctionId);
        console.log("Buyer:", auction.buyer);
        console.log("Max Budget:", auction.maxPrice / 1e6, "USDC");
        console.log("Deadline:", auction.startTime + auction.duration);
        console.log("Service CID:", auction.serviceDescriptionCid);
        console.log("Is Active:", auction.isActive);
        console.log("");
        console.log("The provider agent can now discover this auction via monitor_auctions()");
    }
}
