// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ReverseAuction
 * @dev A reverse auction contract for AI services where buyers preselect eligible providers
 * @author Agentic Marketplace
 */
contract ReverseAuction is ReentrancyGuard {
    using SafeERC20 for IERC20;
    
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
        address[] eligibleProviders;   // Preselected service providers
        address winningProvider;       // Winner of the auction (if any)
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
        uint256 amount;              // Bid amount
        uint256 timestamp;           // When the bid was placed
        uint256 reputation;          // Provider's reputation score (0-100)
        uint256 score;               // Calculated weighted score (0-10000 for precision)
    }
    
    // ============ STATE VARIABLES ============
    
    /// @dev USDC token contract
    IERC20 public immutable USDC_TOKEN;
    
    /// @dev Counter for generating unique auction IDs
    uint256 private _auctionIdCounter;
    
    /// @dev Mapping from auction ID to auction data
    mapping(uint256 => Auction) public auctions;
    
    /// @dev Mapping from auction ID to array of bids
    mapping(uint256 => Bid[]) public auctionBids;
    
    /// @dev Mapping to check if a provider is eligible for a specific auction
    /// auctionId => provider => isEligible
    mapping(uint256 => mapping(address => bool)) public isEligibleProvider;
    
    /// @dev Mapping to track the lowest bid for each auction
    mapping(uint256 => uint256) public lowestBid;
    
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
        address[] eligibleProviders,
        uint256 reputationWeight
    );
    
    event BidPlaced(
        uint256 indexed auctionId,
        address indexed provider,
        uint256 bidAmount,
        uint256 reputation,
        uint256 score,
        uint256 timestamp
    );
    
    event AuctionEnded(
        uint256 indexed auctionId,
        address indexed winner,
        uint256 winningBid
    );
    
    event ServiceCompleted(
        uint256 indexed auctionId,
        address indexed provider
    );
    
    event FundsReleased(
        uint256 indexed auctionId,
        address indexed provider,
        uint256 amount
    );
    
    // ============ ERRORS ============
    
    error InvalidAuctionDuration();
    error InvalidMaxPrice();
    error NoEligibleProviders();
    error InsufficientEscrow();
    error AuctionNotFound();
    error AuctionNotActive();
    error ProviderNotEligible();
    error BidTooHigh();
    error AuctionStillActive();
    error NotAuthorized();
    error InvalidServiceCid();
    error ServiceAlreadyCompleted();
    error InvalidReputationWeight();
    error BidScoreNotCompetitive();
    
    // ============ CONSTRUCTOR ============
    
    /**
     * @dev Constructor sets the USDC token address
     * @param usdcTokenAddress Address of the USDC token contract
     */
    constructor(address usdcTokenAddress) {
        if (usdcTokenAddress == address(0)) revert ();
        USDC_TOKEN = IERC20(usdcTokenAddress);
        _auctionIdCounter = 1; // Start auction IDs from 1
    }
    
    // ============ EXTERNAL FUNCTIONS ============
    
    /**
     * @dev Creates a new reverse auction for an AI service
     * @param serviceDescriptionCid IPFS CID containing service description and requirements
     * @param maxPrice Maximum price buyer is willing to pay (also escrow amount) in USDC
     * @param duration Auction duration in seconds
     * @param eligibleProviders Array of preselected service provider addresses
     * @param reputationWeight Weight for reputation in scoring (0-100, represents 0.00-1.00)
     * @return auctionId The ID of the created auction
     */
    function createAuction(
        string calldata serviceDescriptionCid,
        uint256 maxPrice,
        uint256 duration,
        address[] calldata eligibleProviders,
        uint256 reputationWeight
    ) external nonReentrant returns (uint256 auctionId) {
        // Validation
        if (duration == 0) revert InvalidAuctionDuration();
        if (maxPrice == 0) revert InvalidMaxPrice();
        if (eligibleProviders.length == 0) revert NoEligibleProviders();
        if (bytes(serviceDescriptionCid).length == 0) revert InvalidServiceCid();
        if (reputationWeight > 100) revert InvalidReputationWeight();
        
        // Check USDC allowance and balance
        if (USDC_TOKEN.allowance(msg.sender, address(this)) < maxPrice) revert InsufficientEscrow();
        if (USDC_TOKEN.balanceOf(msg.sender) < maxPrice) revert InsufficientEscrow();
        
        // Generate auction ID
        auctionId = _auctionIdCounter++;
        
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
        auction.eligibleProviders = eligibleProviders;
        auction.winningProvider = address(0);
        auction.winningBid = 0;
        auction.isActive = true;
        auction.isCompleted = false;
        auction.escrowAmount = maxPrice;
        auction.reputationWeight = reputationWeight;
        
        // Set up provider eligibility mapping
        for (uint256 i = 0; i < eligibleProviders.length; i++) {
            isEligibleProvider[auctionId][eligibleProviders[i]] = true;
        }
        
        // Initialize lowest bid to max price
        lowestBid[auctionId] = maxPrice;
        
        // Initialize highest score to 0
        highestScore[auctionId] = 0;
        
        emit AuctionCreated(
            auctionId,
            msg.sender,
            serviceDescriptionCid,
            maxPrice,
            duration,
            eligibleProviders,
            reputationWeight
        );
    }
    
    /**
     * @dev Places a bid in a reverse auction
     * @param auctionId The auction ID to bid on
     * @param bidAmount The bid amount
     * @param reputation The provider's reputation score (0-100)
     */
    function placeBid(
        uint256 auctionId,
        uint256 bidAmount,
        uint256 reputation
    ) external {
        Auction storage auction = auctions[auctionId];
        
        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (!auction.isActive) revert AuctionNotActive();
        if (block.timestamp > auction.startTime + auction.duration) revert AuctionNotActive();
        if (!isEligibleProvider[auctionId][msg.sender]) revert ProviderNotEligible();
        if (bidAmount == 0) revert InvalidMaxPrice(); // Reusing error for zero bid
        if (bidAmount > auction.maxPrice) revert BidTooHigh();
        
        // Calculate weighted score for this bid
        uint256 score = _calculateScore(reputation, bidAmount, auction.maxPrice, auction.reputationWeight);
        
        // Check if this bid's score is better than the current highest score
        // Higher score is better
        if (auction.winningProvider != address(0) && score <= highestScore[auctionId]) {
            revert BidScoreNotCompetitive();
        }
        
        // Create and store the bid
        Bid memory newBid = Bid({
            provider: msg.sender,
            amount: bidAmount,
            timestamp: block.timestamp,
            reputation: reputation,
            score: score
        });
        
        auctionBids[auctionId].push(newBid);
        
        // Update tracking
        lowestBid[auctionId] = bidAmount;
        highestScore[auctionId] = score;
        
        // Update auction state with current best bid
        auction.winningProvider = msg.sender;
        auction.winningBid = bidAmount;
        
        emit BidPlaced(auctionId, msg.sender, bidAmount, reputation, score, block.timestamp);
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
        // If no bids, winningProvider remains address(0)
        address winner = auction.winningProvider;
        uint256 winningAmount = auction.winningBid;
        
        emit AuctionEnded(auctionId, winner, winningAmount);
    }
    
    /**
     * @dev Marks a service as completed and releases payment to the winner
     * @param auctionId The auction ID
     * @dev Can only be called by the buyer to confirm service completion
     */
    function completeService(uint256 auctionId) external nonReentrant {
        Auction storage auction = auctions[auctionId];
        
        // Validation
        if (auction.buyer == address(0)) revert AuctionNotFound();
        if (msg.sender != auction.buyer) revert NotAuthorized();
        if (auction.isActive) revert AuctionStillActive();
        if (auction.isCompleted) revert ServiceAlreadyCompleted();
        if (auction.winningProvider == address(0)) {
            // No winner - this will be handled by refund function
            revert NotAuthorized();
        }
        
        // Mark service as completed
        auction.isCompleted = true;
        
        // Calculate payment amounts
        uint256 winningBid = auction.winningBid;
        uint256 escrowAmount = auction.escrowAmount;
        uint256 refundAmount = escrowAmount - winningBid;
        
        // Reset escrow amount
        auction.escrowAmount = 0;
        
        // Transfer winning bid to service provider
        USDC_TOKEN.safeTransfer(auction.winningProvider, winningBid);
        
        // Refund excess to buyer if any
        if (refundAmount > 0) {
            USDC_TOKEN.safeTransfer(auction.buyer, refundAmount);
        }
        
        emit ServiceCompleted(auctionId, auction.winningProvider);
        emit FundsReleased(auctionId, auction.winningProvider, winningBid);
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
        if (auction.winningProvider != address(0)) revert NotAuthorized(); // Has winner
        if (auction.escrowAmount == 0) revert InsufficientEscrow(); // Already refunded
        
        uint256 refundAmount = auction.escrowAmount;
        auction.escrowAmount = 0;
        auction.isCompleted = true; // Mark as completed to prevent re-entry
        
        // Transfer refund to buyer
        USDC_TOKEN.safeTransfer(auction.buyer, refundAmount);
        
        emit FundsReleased(auctionId, auction.buyer, refundAmount);
    }
    
    // ============ INTERNAL FUNCTIONS ============
    
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
     * @dev Returns the current lowest bid for an auction
     * @param auctionId The auction ID
     * @return currentLowestBid The current lowest bid amount
     */
    function getCurrentLowestBid(uint256 auctionId) external view returns (uint256 currentLowestBid) {
        if (auctions[auctionId].buyer == address(0)) revert AuctionNotFound();
        return lowestBid[auctionId];
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