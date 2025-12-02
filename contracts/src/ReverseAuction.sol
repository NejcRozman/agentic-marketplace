// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import {IERC1271} from "@openzeppelin/contracts/interfaces/IERC1271.sol";
import {IIdentityRegistry} from "./interfaces/IIdentityRegistry.sol";
import {IReputationRegistry} from "./interfaces/IReputationRegistry.sol";

/**
 * @title ReverseAuction
 * @dev A reverse auction contract for AI services integrated with ERC-8004 agent reputation
 * @author Agentic Marketplace
 */
contract ReverseAuction is ReentrancyGuard {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;
    
    // ============ STRUCTS ============
    
    /**
     * @dev Represents a single reverse auction for an AI service
     */
    struct Auction {
        uint256 id;                     // Unique auction identifier
        address buyer;                  // Address of the service buyer
        string serviceDescriptionCid;   // IPFS CID containing service description and requirements
        uint256 maxPrice;              // Maximum price buyer is willing to pay
        uint256 duration;              // Auction duration in seconds
        uint256 startTime;             // When the auction started
        uint256[] eligibleAgentIds;    // Preselected agent IDs from ERC-8004 Identity Registry
        uint256 winningAgentId;        // ID of the winning agent
        uint256 winningBid;           // Winning bid amount
        bool isActive;                // Whether auction is currently active
        bool isCompleted;             // Whether service has been completed
        uint256 escrowAmount;         // Amount held in escrow
        uint256 reputationWeight;     // Weight for reputation (0-100, represents 0.00-1.00)
    }
    
    /**
     * @dev Represents a bid in an auction
     */
    struct Bid {
        address provider;             // Address of the bidding provider
        uint256 agentId;             // ID of the agent used for bidding
        uint256 amount;              // Bid amount
        uint256 timestamp;           // When the bid was placed
        uint256 reputation;          // Agent's reputation score from ERC-8004 (0-100)
        uint256 score;               // Calculated weighted score (0-10000 for precision)
    }
    
    // ============ STATE VARIABLES ============
    
    /// @dev USDC token contract
    IERC20 public immutable USDC_TOKEN;
    
    /// @dev ERC-8004 Identity Registry for agent ownership
    IIdentityRegistry public immutable IDENTITY_REGISTRY;
    
    /// @dev ERC-8004 Reputation Registry for agent reputation
    IReputationRegistry public immutable REPUTATION_REGISTRY;
    
    /// @dev Counter for generating unique auction IDs (public for iteration)
    uint256 public auctionIdCounter;
    
    /// @dev Mapping from auction ID to auction data
    mapping(uint256 => Auction) public auctions;
    
    /// @dev Mapping from auction ID to array of bids
    mapping(uint256 => Bid[]) public auctionBids;
    
    /// @dev Mapping to check if an agent is eligible for a specific auction
    /// auctionId => agentId => isEligible
    mapping(uint256 => mapping(uint256 => bool)) public isEligibleAgent;
    
    /// @dev Mapping to track the winning bid for each auction
    mapping(uint256 => uint256) public winningBid;
    
    /// @dev Mapping to track the highest score for each auction
    mapping(uint256 => uint256) public highestScore;
    
    /// @dev Precision constant for score calculations (2 decimal places)
    uint256 private constant SCORE_PRECISION = 100;
    
    // ============ EVENTS ============
    
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
    
    event AuctionEnded(
        uint256 indexed auctionId,
        uint256 indexed winningAgentId,
        uint256 winningBid
    );
    
    event ServiceCompleted(
        uint256 indexed auctionId,
        uint256 indexed agentId,
        address indexed provider
    );
    
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
    
    // ============ ERRORS ============
    
    error InvalidAuctionDuration();
    error InvalidMaxPrice();
    error NoEligibleAgents();
    error InsufficientEscrow();
    error AuctionNotFound();
    error AuctionNotActive();
    error AgentNotEligible();
    error NotAgentOwner();
    error AgentNotFound();
    error BidTooHigh();
    error AuctionStillActive();
    error NotAuthorized();
    error InvalidServiceCid();
    error ServiceAlreadyCompleted();
    error InvalidReputationWeight();
    error BidScoreNotCompetitive();
    error InvalidFeedbackAuth();
    error FeedbackAuthExpired();
    error InvalidSignature();
    
    // ============ CONSTRUCTOR ============
    
    /**
     * @dev Constructor sets the USDC token and ERC-8004 registry addresses
     * @param usdcTokenAddress Address of the USDC token contract
     * @param identityRegistry Address of the ERC-8004 Identity Registry
     * @param reputationRegistry Address of the ERC-8004 Reputation Registry
     */
    constructor(
        address usdcTokenAddress,
        address identityRegistry,
        address reputationRegistry
    ) {
        if (usdcTokenAddress == address(0)) revert ();
        if (identityRegistry == address(0)) revert ();
        if (reputationRegistry == address(0)) revert ();
        
        USDC_TOKEN = IERC20(usdcTokenAddress);
        IDENTITY_REGISTRY = IIdentityRegistry(identityRegistry);
        REPUTATION_REGISTRY = IReputationRegistry(reputationRegistry);
        auctionIdCounter = 0;
    }
    
    // ============ EXTERNAL FUNCTIONS ============
    
    /**
     * @dev Creates a new reverse auction for an AI service
     * @param serviceDescriptionCid IPFS CID containing service description and requirements
     * @param maxPrice Maximum price buyer is willing to pay (also escrow amount) in USDC
     * @param duration Auction duration in seconds
     * @param eligibleAgentIds Array of preselected agent IDs from ERC-8004 Identity Registry
     * @param reputationWeight Weight for reputation in scoring (0-100, represents 0.00-1.00)
     * @return auctionId The ID of the created auction
     */
    function createAuction(
        string calldata serviceDescriptionCid,
        uint256 maxPrice,
        uint256 duration,
        uint256[] calldata eligibleAgentIds,
        uint256 reputationWeight
    ) external nonReentrant returns (uint256 auctionId) {
        // Validation
        if (duration == 0) revert InvalidAuctionDuration();
        if (maxPrice == 0) revert InvalidMaxPrice();
        if (eligibleAgentIds.length == 0) revert NoEligibleAgents();
        if (bytes(serviceDescriptionCid).length == 0) revert InvalidServiceCid();
        if (reputationWeight > 100) revert InvalidReputationWeight();
        
        // Validate all agents exist in Identity Registry
        for (uint256 i = 0; i < eligibleAgentIds.length; i++) {
            if (!_agentExists(eligibleAgentIds[i])) revert AgentNotFound();
        }
        
        // Check USDC allowance and balance
        if (USDC_TOKEN.allowance(msg.sender, address(this)) < maxPrice) revert InsufficientEscrow();
        if (USDC_TOKEN.balanceOf(msg.sender) < maxPrice) revert InsufficientEscrow();
        
        // Generate auction ID
        auctionId = ++auctionIdCounter;
        
        // Transfer USDC to escrow
        USDC_TOKEN.safeTransferFrom(msg.sender, address(this), maxPrice);
        
        // Create auction
        Auction storage auction = auctions[auctionId];
        auction.id = auctionId;
        auction.buyer = msg.sender;
        auction.serviceDescriptionCid = serviceDescriptionCid;
        auction.maxPrice = maxPrice;
        auction.duration = duration;
        auction.startTime = block.timestamp;
        auction.eligibleAgentIds = eligibleAgentIds;
        auction.winningAgentId = 0;
        auction.winningBid = 0;
        auction.isActive = true;
        auction.isCompleted = false;
        auction.escrowAmount = maxPrice;
        auction.reputationWeight = reputationWeight;
        
        // Set up agent eligibility mapping for O(1) lookup
        for (uint256 i = 0; i < eligibleAgentIds.length; i++) {
            isEligibleAgent[auctionId][eligibleAgentIds[i]] = true;
        }
        
        // Initialize winning bid to max price
        winningBid[auctionId] = maxPrice;
        
        // Initialize highest score to 0
        highestScore[auctionId] = 0;
        
        emit AuctionCreated(
            auctionId,
            msg.sender,
            serviceDescriptionCid,
            maxPrice,
            duration,
            eligibleAgentIds,
            reputationWeight
        );
    }
    
    /**
     * @dev Places a bid in a reverse auction
     * @param auctionId The auction ID to bid on
     * @param bidAmount The bid amount
     * @param agentId The agent ID to bid with
     */
    function placeBid(
        uint256 auctionId,
        uint256 bidAmount,
        uint256 agentId
    ) external {
        Auction storage auction = auctions[auctionId];
        
        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) revert AuctionNotActive();
        if (block.timestamp > auction.startTime + auction.duration) revert AuctionNotActive();
        if (!isEligibleAgent[auctionId][agentId]) revert AgentNotEligible();
        if (bidAmount == 0) revert InvalidMaxPrice(); // Reusing error for zero bid
        if (bidAmount > auction.maxPrice) revert BidTooHigh();
        
        // Check caller owns the agent
        if (IDENTITY_REGISTRY.ownerOf(agentId) != msg.sender) revert NotAgentOwner();
        
        // Fetch reputation from ERC-8004 Reputation Registry
        address[] memory emptyAddresses = new address[](0);
        (uint256 feedbackCount, uint256 averageScore) = REPUTATION_REGISTRY.getSummary(
            agentId,
            emptyAddresses,
            0,  // tag1 - no filter
            0   // tag2 - no filter
        );
        
        // Use default reputation of 50 if agent has no feedback
        uint256 reputation = feedbackCount > 0 ? averageScore : 50;
        
        // Calculate weighted score for this bid
        uint256 score = _calculateScore(reputation, bidAmount, auction.maxPrice, auction.reputationWeight);
        
        // Check if this bid's score is better than the current highest score
        // Higher score is better
        if (auction.winningAgentId != 0 && score <= highestScore[auctionId]) {
            revert BidScoreNotCompetitive();
        }
        
        // Create and store the bid
        Bid memory newBid = Bid({
            provider: msg.sender,
            agentId: agentId,
            amount: bidAmount,
            timestamp: block.timestamp,
            reputation: reputation,
            score: score
        });
        
        auctionBids[auctionId].push(newBid);
        
        // Update tracking
        winningBid[auctionId] = bidAmount;
        highestScore[auctionId] = score;
        
        // Update auction state with current best bid
        auction.winningAgentId = agentId;
        auction.winningBid = bidAmount;
        
        emit BidPlaced(auctionId, msg.sender, agentId, bidAmount, reputation, score, block.timestamp);
    }
    
    /**
     * @dev Ends an auction and finalizes the winner
     * @param auctionId The auction ID to end
     * @dev Can be called by anyone once auction time has expired, or by buyer at any time
     */
    function endAuction(uint256 auctionId) external {
        Auction storage auction = auctions[auctionId];
        
        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) revert AuctionNotActive();
        
        // Check who can end the auction
        bool timeExpired = block.timestamp > auction.startTime + auction.duration;
        bool isBuyer = msg.sender == auction.buyer;
        
        if (!timeExpired && !isBuyer) revert NotAuthorized();
        
        // Deactivate the auction
        auction.isActive = false;
        
        // If there are bids, the current winner is already set
        // If no bids, winningAgentId remains 0
        uint256 winningAgent = auction.winningAgentId;
        uint256 winningAmount = auction.winningBid;
        
        emit AuctionEnded(auctionId, winningAgent, winningAmount);
    }
    
    /**
     * @dev Marks a service as completed and releases payment to the winner
     * @param auctionId The auction ID
     * @param feedbackAuth Signed authorization for buyer to give feedback (per ERC-8004)
     * @dev Can only be called by the service provider (owner of winning agent)
     * 
     * The feedbackAuth must be in ERC-8004 format:
     * - First 224 bytes: abi.encode(agentId, clientAddress, indexLimit, expiry, chainId, identityRegistry, signerAddress)
     * - Last 65 bytes: signature
     * 
     * This allows the buyer to later submit feedback to the ERC-8004 Reputation Registry.
     */
    function completeService(uint256 auctionId, bytes calldata feedbackAuth) external nonReentrant {
        Auction storage auction = auctions[auctionId];
        
        // Basic validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (auction.isActive) revert AuctionStillActive();
        if (auction.isCompleted) revert ServiceAlreadyCompleted();
        if (auction.winningAgentId == 0) revert NotAuthorized();
        
        // Check caller is agent owner
        address currentOwner = IDENTITY_REGISTRY.ownerOf(auction.winningAgentId);
        if (msg.sender != currentOwner) revert NotAgentOwner();
        
        // Verify feedbackAuth is valid and properly signed
        _verifyFeedbackAuth(
            feedbackAuth,
            auction.winningAgentId,
            auction.buyer,
            currentOwner
        );
        
        // Release payment
        auction.isCompleted = true;
        
        uint256 winningBid = auction.winningBid;
        uint256 escrowAmount = auction.escrowAmount;
        uint256 refundAmount = escrowAmount - winningBid;
        
        auction.escrowAmount = 0;
        
        // Transfer winning bid to provider
        USDC_TOKEN.safeTransfer(currentOwner, winningBid);
        
        // Refund excess to buyer
        if (refundAmount > 0) {
            USDC_TOKEN.safeTransfer(auction.buyer, refundAmount);
        }
        
        // ===== EMIT EVENTS =====
        emit ServiceCompleted(auctionId, auction.winningAgentId, currentOwner);
        emit FundsReleased(auctionId, auction.winningAgentId, currentOwner, winningBid);
        
        // Emit feedbackAuth for buyer to retrieve from event logs
        // Buyer can later use this to submit feedback to ERC-8004 Reputation Registry
        emit FeedbackAuthProvided(
            auctionId,
            auction.winningAgentId,
            auction.buyer,
            feedbackAuth
        );
    }
    
    /**
     * @dev Allows buyer to get refund if auction ended with no bids
     * @param auctionId The auction ID
     */
    function refundBuyer(uint256 auctionId) external nonReentrant {
        Auction storage auction = auctions[auctionId];
        
        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (msg.sender != auction.buyer) revert NotAuthorized();
        if (auction.isActive) revert AuctionStillActive();
        if (auction.isCompleted) revert ServiceAlreadyCompleted();
        if (auction.winningAgentId != 0) revert NotAuthorized(); // Has winner
        if (auction.escrowAmount == 0) revert InsufficientEscrow(); // Already refunded
        
        uint256 refundAmount = auction.escrowAmount;
        auction.escrowAmount = 0;
        auction.isCompleted = true; // Mark as completed to prevent re-entry
        
        // Transfer refund to buyer
        USDC_TOKEN.safeTransfer(auction.buyer, refundAmount);
        
        emit FundsReleased(auctionId, 0, auction.buyer, refundAmount);
    }
    
    // ============ INTERNAL FUNCTIONS ============
    
    /**
     * @dev Checks if an agent exists in the Identity Registry
     * @param agentId The agent token ID to check
     * @return exists True if the agent has been minted and exists
     */
    function _agentExists(uint256 agentId) internal view returns (bool exists) {
        try IDENTITY_REGISTRY.ownerOf(agentId) returns (address owner) {
            return owner != address(0);
        } catch {
            return false;
        }
    }
    
    /**
     * @dev Calculates the weighted score for a bid
     * @param reputation Provider's reputation score (0-100)
     * @param bidAmount The bid amount
     * @param maxPrice Maximum price from auction
     * @param reputationWeight Weight for reputation (0-100, represents 0.00-1.00)
     * @return score The calculated weighted score (0-10000)
     * 
     * Formula: score = w * normalize(reputation) + (1 - w) * normalize(1 / bidAmount)
     * where:
     *   - w = reputationWeight / 100
     *   - normalize(reputation) = reputation / 100
     *   - normalize(1 / bidAmount) = 1 - (bidAmount / maxPrice)
     * 
     * Higher score is better
     */
    function _calculateScore(
        uint256 reputation,
        uint256 bidAmount,
        uint256 maxPrice,
        uint256 reputationWeight
    ) internal pure returns (uint256 score) {
        // Normalize reputation: reputation is already 0-100, so we just use it
        // For precision, we multiply by SCORE_PRECISION (100)
        uint256 normalizedReputation = reputation; // Already 0-100
        
        // Normalize bid amount (inverted so lower price = higher score)
        // Formula: (1 - bidAmount/maxPrice) * 100
        // To avoid division precision issues: (maxPrice - bidAmount) * 100 / maxPrice
        uint256 normalizedBidScore = ((maxPrice - bidAmount) * SCORE_PRECISION) / maxPrice;
        
        // Calculate weighted score
        // score = w * normalizedReputation + (100 - w) * normalizedBidScore
        // All values are already scaled by 100, so result is in range 0-10000
        score = (reputationWeight * normalizedReputation + 
                (SCORE_PRECISION - reputationWeight) * normalizedBidScore) / SCORE_PRECISION;
        
        return score;
    }
    
    /**
     * @dev Verifies feedbackAuth signature and parameters (ERC-8004 compatible)
     * @param feedbackAuth The encoded feedbackAuth from provider
     * @param expectedAgentId The agent ID that should be in feedbackAuth
     * @param expectedClient The buyer address that should be in feedbackAuth
     * @param currentOwner The current owner of the agent (for authorization check)
     */
    function _verifyFeedbackAuth(
        bytes calldata feedbackAuth,
        uint256 expectedAgentId,
        address expectedClient,
        address currentOwner
    ) internal view {
        // ERC-8004 format: [224 bytes params][65 bytes signature]
        if (feedbackAuth.length < 289) revert InvalidFeedbackAuth();
        
        // Decode first 224 bytes
        (
            uint256 authAgentId,
            address authClient,
            uint64 authIndexLimit,
            uint256 authExpiry,
            uint256 authChainId,
            address authIdentityRegistry,
            address authSigner
        ) = abi.decode(feedbackAuth[:224], (uint256, address, uint64, uint256, uint256, address, address));
        
        // Extract signature (last 65 bytes)
        bytes memory signature = feedbackAuth[224:];
        
        // Verify tuple fields match expected context
        if (authAgentId != expectedAgentId) revert InvalidFeedbackAuth();
        if (authClient != expectedClient) revert InvalidFeedbackAuth();
        if (authExpiry <= block.timestamp) revert FeedbackAuthExpired();
        if (authChainId != block.chainid) revert InvalidFeedbackAuth();
        if (authIdentityRegistry != address(IDENTITY_REGISTRY)) revert InvalidFeedbackAuth();
        
        // Reconstruct message hash and apply EIP-191 prefix (matching official ERC-8004)
        bytes32 messageHash = MessageHashUtils.toEthSignedMessageHash(
            keccak256(
                abi.encode(
                    authAgentId,
                    authClient,
                    authIndexLimit,
                    authExpiry,
                    authChainId,
                    authIdentityRegistry,
                    authSigner
                )
            )
        );
        
        // Verify signature (supports both EOA and ERC-1271 smart contract wallets)
        address recoveredSigner = ECDSA.recover(messageHash, signature);
        if (recoveredSigner != authSigner) {
            // If not EOA signature, try ERC-1271 for smart contract wallets
            if (authSigner.code.length > 0) {
                // Try ERC-1271 verification
                try IERC1271(authSigner).isValidSignature(messageHash, signature) returns (bytes4 magicValue) {
                    if (magicValue != IERC1271.isValidSignature.selector) {
                        revert InvalidSignature();
                    }
                } catch {
                    revert InvalidSignature();
                }
            } else {
                revert InvalidSignature();
            }
        }
        
        // Verify signer is authorized (owner or approved operator)
        if (authSigner != currentOwner) {
            // Check if signer is an approved operator
            bool isApproved = IDENTITY_REGISTRY.isApprovedForAll(currentOwner, authSigner) ||
                             IDENTITY_REGISTRY.getApproved(expectedAgentId) == authSigner;
            if (!isApproved) revert NotAgentOwner();
        }
    }
    
    // ============ VIEW FUNCTIONS ============
    
    /**
     * @dev Returns the end time of an auction
     * @param auctionId The auction ID
     * @return endTime When the auction ends (startTime + duration)
     */
    function getAuctionEndTime(uint256 auctionId) external view returns (uint256 endTime) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        return auction.startTime + auction.duration;
    }
    
    /**
     * @dev Checks if an auction is currently active
     * @param auctionId The auction ID
     * @return active True if auction is active and not expired
     */
    function isAuctionActive(uint256 auctionId) external view returns (bool active) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        
        return auction.isActive && (block.timestamp <= auction.startTime + auction.duration);
    }
    
    /**
     * @dev Returns the current winning bid for an auction
     * @param auctionId The auction ID
     * @return currentWinningBid The current winning bid amount
     */
    function getCurrentWinningBid(uint256 auctionId) external view returns (uint256 currentWinningBid) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return winningBid[auctionId];
    }
    
    /**
     * @dev Returns the current highest score for an auction
     * @param auctionId The auction ID
     * @return currentHighestScore The current highest weighted score
     */
    function getCurrentHighestScore(uint256 auctionId) external view returns (uint256 currentHighestScore) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return highestScore[auctionId];
    }
    
    /**
     * @dev Returns all bids for a specific auction
     * @param auctionId The auction ID
     * @return bids Array of all bids placed on the auction
     */
    function getAuctionBids(uint256 auctionId) external view returns (Bid[] memory bids) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return auctionBids[auctionId];
    }
    
    /**
     * @dev Returns the number of bids placed on an auction
     * @param auctionId The auction ID
     * @return bidCount Total number of bids
     */
    function getBidCount(uint256 auctionId) external view returns (uint256 bidCount) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return auctionBids[auctionId].length;
    }
    
    /**
     * @dev Returns the time remaining in an auction
     * @param auctionId The auction ID
     * @return timeRemaining Seconds remaining (0 if expired)
     */
    function getTimeRemaining(uint256 auctionId) external view returns (uint256 timeRemaining) {
        Auction storage auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        
        uint256 endTime = auction.startTime + auction.duration;
        if (block.timestamp >= endTime) {
            return 0;
        }
        return endTime - block.timestamp;
    }
    
    /**
     * @dev Returns complete auction information
     * @param auctionId The auction ID
     * @return auction The complete auction struct
     */
    function getAuctionDetails(uint256 auctionId) external view returns (Auction memory auction) {
        auction = auctions[auctionId];
        if (auction.buyer == address(0)) revert AuctionNotFound();
        return auction;
    }
}