// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console2 as console} from "forge-std/Test.sol";
import {ReverseAuction} from "../src/ReverseAuction.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {ERC20Mock} from "@openzeppelin/contracts/mocks/token/ERC20Mock.sol";
import {IIdentityRegistry} from "../src/interfaces/IIdentityRegistry.sol";
import {IReputationRegistry} from "../src/interfaces/IReputationRegistry.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

contract MockIdentityRegistry is IIdentityRegistry {
    mapping(uint256 => address) private _owners;
    uint256 private _tokenIdCounter;

    function register() external returns (uint256) {
        uint256 tokenId = ++_tokenIdCounter;
        _owners[tokenId] = msg.sender;
        return tokenId;
    }

    function ownerOf(uint256 tokenId) external view override returns (address) {
        address owner = _owners[tokenId];
        require(owner != address(0), "Token does not exist");
        return owner;
    }

    function exists(uint256 tokenId) external view override returns (bool) {
        return _owners[tokenId] != address(0);
    }

    function setOwner(uint256 tokenId, address newOwner) external {
        _owners[tokenId] = newOwner;
    }

    // ERC721 required functions (minimal implementation)
    function balanceOf(address) external pure returns (uint256) {
        return 0;
    }

    function safeTransferFrom(address, address, uint256) external pure {
        revert("Not implemented");
    }

    function safeTransferFrom(address, address, uint256, bytes memory) external pure {
        revert("Not implemented");
    }

    function transferFrom(address, address, uint256) external pure {
        revert("Not implemented");
    }

    function approve(address, uint256) external pure {
        revert("Not implemented");
    }

    function setApprovalForAll(address, bool) external pure {
        revert("Not implemented");
    }

    function getApproved(uint256) external pure returns (address) {
        return address(0);
    }

    function isApprovedForAll(address, address) external pure returns (bool) {
        return false;
    }

    function supportsInterface(bytes4) external pure returns (bool) {
        return true;
    }
}

contract MockReputationRegistry is IReputationRegistry {
    mapping(uint256 => mapping(address => uint256)) private _feedbackCounts;
    mapping(uint256 => mapping(address => uint256)) private _scores;

    function setReputation(uint256 agentId, address client, uint256 count, uint256 score) external {
        _feedbackCounts[agentId][client] = count;
        _scores[agentId][client] = score;
    }

    function getSummary(
        uint256 agentId,
        address[] calldata clientAddresses,
        uint256,
        uint256
    ) external view override returns (uint256 feedbackCount, uint256 averageScore) {
        if (clientAddresses.length == 0) {
            // Return default for all clients
            feedbackCount = _feedbackCounts[agentId][address(0)];
            averageScore = _scores[agentId][address(0)];
        } else {
            // Return for specific client
            feedbackCount = _feedbackCounts[agentId][clientAddresses[0]];
            averageScore = _scores[agentId][clientAddresses[0]];
        }
    }
}

contract ReverseAuctionTest is Test {
    ReverseAuction public auction;
    ERC20Mock public usdc;
    MockIdentityRegistry public identityRegistry;
    MockReputationRegistry public reputationRegistry;

    address public buyer;
    address public provider1;
    address public provider2;
    address public provider3;
    
    uint256 public buyerPk = 1;
    uint256 public provider1Pk = 2;
    uint256 public provider2Pk = 3;
    uint256 public provider3Pk = 4;

    uint256 public agentId1;
    uint256 public agentId2;
    uint256 public agentId3;

    uint256 constant MAX_PRICE = 1000e6; // 1000 USDC
    uint256 constant DURATION = 7 days;
    uint256 constant REPUTATION_WEIGHT = 50; // 0.50

    event AuctionCreated(
        uint256 indexed auctionId,
        address indexed buyer,
        string serviceDescriptionCid,
        uint256 maxPrice,
        uint256 duration,
        uint256[] eligibleAgentIds,
        uint256 reputationWeight
    );

    event BidPlaced(
        uint256 indexed auctionId,
        address indexed provider,
        uint256 indexed agentId,
        uint256 bidAmount,
        uint256 reputation,
        uint256 score,
        uint256 timestamp
    );

    event AuctionEnded(uint256 indexed auctionId, uint256 indexed winningAgentId, uint256 winningBid);

    event ServiceCompleted(uint256 indexed auctionId, uint256 indexed agentId, address indexed provider);

    event FundsReleased(
        uint256 indexed auctionId,
        uint256 indexed agentId,
        address indexed provider,
        uint256 amount
    );

    event FeedbackAuthProvided(
        uint256 indexed auctionId,
        uint256 indexed agentId,
        address indexed buyer,
        bytes feedbackAuth
    );

    function setUp() public {
        // Setup addresses from private keys
        buyer = vm.addr(buyerPk);
        provider1 = vm.addr(provider1Pk);
        provider2 = vm.addr(provider2Pk);
        provider3 = vm.addr(provider3Pk);
        
        // Deploy mocks
        usdc = new ERC20Mock();
        identityRegistry = new MockIdentityRegistry();
        reputationRegistry = new MockReputationRegistry();

        // Deploy auction contract
        auction = new ReverseAuction(address(usdc), address(identityRegistry), address(reputationRegistry));

        // Setup agents
        vm.startPrank(provider1);
        agentId1 = identityRegistry.register();
        vm.stopPrank();

        vm.startPrank(provider2);
        agentId2 = identityRegistry.register();
        vm.stopPrank();

        vm.startPrank(provider3);
        agentId3 = identityRegistry.register();
        vm.stopPrank();

        // Setup reputations
        reputationRegistry.setReputation(agentId1, address(0), 10, 80); // 80/100
        reputationRegistry.setReputation(agentId2, address(0), 5, 60); // 60/100
        reputationRegistry.setReputation(agentId3, address(0), 0, 0); // No reputation

        // Give buyer USDC
        usdc.mint(buyer, 10000e6);
    }

    function testDebugSignatureRecovery() public view {
        // Test that our signing mechanism works correctly
        uint256 testPk = provider1Pk;
        address expectedAddr = provider1;
        
        console.log("Testing signature recovery:");
        console.log("Private key:", testPk);
        console.log("Expected address:", expectedAddr);
        console.log("vm.addr(pk):", vm.addr(testPk));
        
        // Create a simple message
        bytes32 messageHash = keccak256(abi.encodePacked("test message"));
        console.log("Message hash:");
        console.logBytes32(messageHash);
        
        // Apply EIP-191 prefix
        bytes32 ethSignedMessageHash = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", messageHash)
        );
        console.log("ETH signed message hash:");
        console.logBytes32(ethSignedMessageHash);
        
        // Sign it
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(testPk, ethSignedMessageHash);
        bytes memory signature = abi.encodePacked(r, s, v);
        console.log("Signature v:", v);
        console.logBytes32(r);
        console.logBytes32(s);
        
        // Try to recover using ECDSA
        address recoveredAddr = ECDSA.recover(ethSignedMessageHash, signature);
        console.log("Recovered address:", recoveredAddr);
        console.log("Match:", recoveredAddr == expectedAddr);
        
        require(recoveredAddr == expectedAddr, "Signature recovery failed");
    }

    function testCreateAuction() public {
        uint256[] memory eligibleAgents = new uint256[](2);
        eligibleAgents[0] = agentId1;
        eligibleAgents[1] = agentId2;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectEmit(true, true, false, true);
        emit AuctionCreated(1, buyer, "QmTest", MAX_PRICE, DURATION, eligibleAgents, REPUTATION_WEIGHT);

        uint256 auctionId = auction.createAuction("QmTest", MAX_PRICE, DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();

        assertEq(auctionId, 1);
        assertEq(usdc.balanceOf(address(auction)), MAX_PRICE);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.buyer, buyer);
        assertEq(auc.maxPrice, MAX_PRICE);
        assertEq(auc.duration, DURATION);
        assertEq(auc.isActive, true);
        assertEq(auc.isCompleted, false);
    }

    function testCreateAuctionRevertsWithInvalidDuration() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId1;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.InvalidAuctionDuration.selector);
        auction.createAuction("QmTest", MAX_PRICE, 0, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithInvalidMaxPrice() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId1;

        vm.startPrank(buyer);
        vm.expectRevert(ReverseAuction.InvalidMaxPrice.selector);
        auction.createAuction("QmTest", 0, DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithNoEligibleAgents() public {
        uint256[] memory eligibleAgents = new uint256[](0);

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.NoEligibleAgents.selector);
        auction.createAuction("QmTest", MAX_PRICE, DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testCreateAuctionRevertsWithNonExistentAgent() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = 999; // Non-existent agent

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);

        vm.expectRevert(ReverseAuction.AgentNotFound.selector);
        auction.createAuction("QmTest", MAX_PRICE, DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();
    }

    function testPlaceBid() public {
        uint256 auctionId = _createTestAuction();

        uint256 bidAmount = 800e6;

        vm.prank(provider1);
        vm.expectEmit(true, true, true, false); // Don't check data for timestamp
        emit BidPlaced(auctionId, provider1, agentId1, bidAmount, 80, 7000, block.timestamp);

        auction.placeBid(auctionId, bidAmount, agentId1);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.winningAgentId, agentId1);
        assertEq(auc.winningBid, bidAmount);
    }

    function testPlaceBidWithBetterScore() public {
        uint256 auctionId = _createTestAuction();

        // First bid: provider2 with lower reputation (60) but lower price
        vm.prank(provider2);
        auction.placeBid(auctionId, 700e6, agentId2);

        // Second bid: provider1 with higher reputation (80) and slightly higher price
        vm.prank(provider1);
        auction.placeBid(auctionId, 750e6, agentId1);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.winningAgentId, agentId1); // Should win with better overall score
    }

    function testPlaceBidRevertsWhenNotCompetitive() public {
        uint256 auctionId = _createTestAuction();

        // First bid with excellent score
        vm.prank(provider1);
        auction.placeBid(auctionId, 600e6, agentId1);

        // Second bid with worse score should revert
        vm.prank(provider2);
        vm.expectRevert(ReverseAuction.BidScoreNotCompetitive.selector);
        auction.placeBid(auctionId, 900e6, agentId2);
    }

    function testPlaceBidRevertsWhenNotAgentOwner() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider2); // Not owner of agentId1
        vm.expectRevert(ReverseAuction.NotAgentOwner.selector);
        auction.placeBid(auctionId, 800e6, agentId1);
    }

    function testPlaceBidRevertsWhenAgentNotEligible() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider3);
        vm.expectRevert(ReverseAuction.AgentNotEligible.selector);
        auction.placeBid(auctionId, 800e6, agentId3); // agentId3 not in eligible list
    }

    function testPlaceBidRevertsWhenBidTooHigh() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.BidTooHigh.selector);
        auction.placeBid(auctionId, MAX_PRICE + 1, agentId1);
    }

    function testPlaceBidWithDefaultReputation() public {
        uint256[] memory eligibleAgents = new uint256[](1);
        eligibleAgents[0] = agentId3; // Agent with no reputation

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);
        uint256 auctionId = auction.createAuction("QmTest", MAX_PRICE, DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();

        vm.prank(provider3);
        auction.placeBid(auctionId, 800e6, agentId3);

        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        assertEq(bids[0].reputation, 50); // Default reputation
    }

    function testEndAuction() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        // Fast forward past auction end
        vm.warp(block.timestamp + DURATION + 1);

        vm.expectEmit(true, true, false, true);
        emit AuctionEnded(auctionId, agentId1, 800e6);

        auction.endAuction(auctionId);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isActive, false);
    }

    function testEndAuctionByBuyerEarly() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        // Buyer can end early
        vm.prank(buyer);
        auction.endAuction(auctionId);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isActive, false);
    }

    function testEndAuctionRevertsWhenNotAuthorized() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        // Random user cannot end auction before time expires
        vm.prank(address(999));
        vm.expectRevert(ReverseAuction.NotAuthorized.selector);
        auction.endAuction(auctionId);
    }

    function testCompleteService() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        // Create feedbackAuth with proper private key
        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk);

        uint256 providerBalanceBefore = usdc.balanceOf(provider1);
        uint256 buyerBalanceBefore = usdc.balanceOf(buyer);

        vm.prank(provider1);
        auction.completeService(auctionId, feedbackAuth);

        assertEq(usdc.balanceOf(provider1), providerBalanceBefore + 800e6);
        assertEq(usdc.balanceOf(buyer), buyerBalanceBefore + 200e6); // Refund of excess

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isCompleted, true);
        assertEq(auc.escrowAmount, 0);
    }

    function testCompleteServiceRevertsWhenNotAgentOwner() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk);

        vm.prank(provider2); // Not the owner
        vm.expectRevert(ReverseAuction.NotAgentOwner.selector);
        auction.completeService(auctionId, feedbackAuth);
    }

    function testCompleteServiceRevertsWithInvalidFeedbackAuth() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        // Create feedbackAuth with wrong agentId
        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId2, buyer, provider1, provider1Pk);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.InvalidFeedbackAuth.selector);
        auction.completeService(auctionId, feedbackAuth);
    }

    function testCompleteServiceRevertsWithExpiredFeedbackAuth() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        // Create feedbackAuth that's already expired
        uint256 expiry = block.timestamp - 1;
        bytes memory feedbackAuth =
            _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk, expiry);

        vm.prank(provider1);
        vm.expectRevert(ReverseAuction.FeedbackAuthExpired.selector);
        auction.completeService(auctionId, feedbackAuth);
    }

    function testCompleteServiceHandlesAgentTransfer() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        // Transfer agent to new owner
        address newOwner = address(999);
        usdc.mint(newOwner, 1000e6); // Give new owner some tokens for test
        identityRegistry.setOwner(agentId1, newOwner);

        // Use vm.addr to get the correct address for private key 999
        uint256 newOwnerPk = 999;
        address newOwnerAddr = vm.addr(newOwnerPk);
        identityRegistry.setOwner(agentId1, newOwnerAddr);

        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, newOwnerAddr, newOwnerPk);

        uint256 newOwnerBalanceBefore = usdc.balanceOf(newOwnerAddr);

        vm.prank(newOwnerAddr);
        auction.completeService(auctionId, feedbackAuth);

        // Payment goes to current owner, not original bidder
        assertEq(usdc.balanceOf(newOwnerAddr), newOwnerBalanceBefore + 800e6);
    }

    function testRefundBuyer() public {
        uint256 auctionId = _createTestAuction();

        // No bids placed
        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        uint256 buyerBalanceBefore = usdc.balanceOf(buyer);

        vm.prank(buyer);
        auction.refundBuyer(auctionId);

        assertEq(usdc.balanceOf(buyer), buyerBalanceBefore + MAX_PRICE);

        ReverseAuction.Auction memory auc = auction.getAuctionDetails(auctionId);
        assertEq(auc.isCompleted, true);
        assertEq(auc.escrowAmount, 0);
    }

    function testRefundBuyerRevertsWhenHasWinner() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        vm.prank(buyer);
        vm.expectRevert(ReverseAuction.NotAuthorized.selector);
        auction.refundBuyer(auctionId);
    }

    function testGetAuctionBids() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider2);
        auction.placeBid(auctionId, 900e6, agentId2);

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        assertEq(bids.length, 2);
        assertEq(bids[0].provider, provider2);
        assertEq(bids[1].provider, provider1);
    }

    function testGetBidCount() public {
        uint256 auctionId = _createTestAuction();

        assertEq(auction.getBidCount(auctionId), 0);

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        assertEq(auction.getBidCount(auctionId), 1);
    }

    function testGetTimeRemaining() public {
        uint256 auctionId = _createTestAuction();

        uint256 timeRemaining = auction.getTimeRemaining(auctionId);
        assertEq(timeRemaining, DURATION);

        vm.warp(block.timestamp + 3 days);
        timeRemaining = auction.getTimeRemaining(auctionId);
        assertEq(timeRemaining, 4 days);

        vm.warp(block.timestamp + 5 days);
        timeRemaining = auction.getTimeRemaining(auctionId);
        assertEq(timeRemaining, 0);
    }

    function testIsAuctionActive() public {
        uint256 auctionId = _createTestAuction();

        assertTrue(auction.isAuctionActive(auctionId));

        vm.warp(block.timestamp + DURATION + 1);
        assertFalse(auction.isAuctionActive(auctionId));
    }

    function testGetCurrentWinningBid() public {
        uint256 auctionId = _createTestAuction();

        assertEq(auction.getCurrentWinningBid(auctionId), MAX_PRICE);

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        assertEq(auction.getCurrentWinningBid(auctionId), 800e6);
    }

    function testScoreCalculation() public {
        uint256 auctionId = _createTestAuction();

        // Test that higher reputation + lower price = higher score
        vm.prank(provider1);
        auction.placeBid(auctionId, 700e6, agentId1); // rep: 80, price: 700

        ReverseAuction.Bid[] memory bids = auction.getAuctionBids(auctionId);
        uint256 score1 = bids[0].score;

        // Create new auction for comparison
        uint256 auctionId2 = _createTestAuction();

        vm.prank(provider2);
        auction.placeBid(auctionId2, 800e6, agentId2); // rep: 60, price: 800

        bids = auction.getAuctionBids(auctionId2);
        uint256 score2 = bids[0].score;

        // Higher reputation and lower price should give higher score
        assertGt(score1, score2);
    }

    function testFeedbackAuthEvent() public {
        uint256 auctionId = _createTestAuction();

        vm.prank(provider1);
        auction.placeBid(auctionId, 800e6, agentId1);

        vm.warp(block.timestamp + DURATION + 1);
        auction.endAuction(auctionId);

        bytes memory feedbackAuth = _createFeedbackAuthSigned(auctionId, agentId1, buyer, provider1, provider1Pk);

        vm.prank(provider1);
        // Just check the indexed event parameters, skip the feedbackAuth bytes data
        vm.expectEmit(true, true, true, false);
        emit FeedbackAuthProvided(auctionId, agentId1, buyer, feedbackAuth);

        auction.completeService(auctionId, feedbackAuth);
    }

    // Helper functions

    function _createTestAuction() internal returns (uint256) {
        uint256[] memory eligibleAgents = new uint256[](2);
        eligibleAgents[0] = agentId1;
        eligibleAgents[1] = agentId2;

        vm.startPrank(buyer);
        usdc.approve(address(auction), MAX_PRICE);
        uint256 auctionId = auction.createAuction("QmTest", MAX_PRICE, DURATION, eligibleAgents, REPUTATION_WEIGHT);
        vm.stopPrank();

        return auctionId;
    }

    function _createFeedbackAuth(uint256 auctionId, uint256 agentId, address client, address signer)
        internal
        view
        returns (bytes memory)
    {
        return _createFeedbackAuthWithExpiry(auctionId, agentId, client, signer, block.timestamp + 30 days);
    }

    function _createFeedbackAuthWithExpiry(
        uint256, /* auctionId */
        uint256 agentId,
        address client,
        address signer,
        uint256 expiry
    ) internal view returns (bytes memory) {
        // Get private key for signer (test helper)
        uint256 signerPrivateKey = _getPrivateKey(signer);
        return _createFeedbackAuthSigned(0, agentId, client, signer, signerPrivateKey, expiry);
    }

    function _createFeedbackAuthSigned(
        uint256, /* auctionId */
        uint256 agentId,
        address client,
        address signer,
        uint256 signerPrivateKey
    ) internal view returns (bytes memory) {
        return _createFeedbackAuthSigned(0, agentId, client, signer, signerPrivateKey, block.timestamp + 30 days);
    }

    function _createFeedbackAuthSigned(
        uint256, /* auctionId */
        uint256 agentId,
        address client,
        address signer,
        uint256 signerPrivateKey,
        uint256 expiry
    ) internal view returns (bytes memory) {
        uint64 indexLimit = 1;
        uint256 chainId = block.chainid;
        address identityReg = address(identityRegistry);

        // Create message hash using abi.encode (matching ERC-8004)
        bytes32 rawHash = keccak256(
            abi.encode(agentId, client, indexLimit, expiry, chainId, identityReg, signer)
        );

        // Apply EIP-191 prefix using MessageHashUtils (matching official implementation)
        bytes32 messageHash = MessageHashUtils.toEthSignedMessageHash(rawHash);

        // Sign the prefixed hash
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(signerPrivateKey, messageHash);
        bytes memory signature = abi.encodePacked(r, s, v);

        // Encode in ERC-8004 format: [224 bytes params][65 bytes signature]
        bytes memory params = abi.encode(agentId, client, indexLimit, expiry, chainId, identityReg, signer);
        return abi.encodePacked(params, signature);
    }

    function _getPrivateKey(address addr) internal view returns (uint256) {
        // Map test addresses to private keys
        if (addr == buyer) return buyerPk;
        if (addr == provider1) return provider1Pk;
        if (addr == provider2) return provider2Pk;
        if (addr == provider3) return provider3Pk;
        return 1;
    }
}
