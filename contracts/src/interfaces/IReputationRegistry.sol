// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IReputationRegistry
 * @dev Interface for ERC-8004 Reputation Registry
 * @notice Manages reputation feedback for agents with scores from 0-100
 */
interface IReputationRegistry {
    /**
     * @dev Give feedback for an agent (ERC-8004 standard)
     * @param agentId The agent token ID to give feedback for
     * @param score The reputation score (0-100, REQUIRED)
     * @param tag1 Optional first tag for categorization (use bytes32(0) for none)
     * @param tag2 Optional second tag for categorization (use bytes32(0) for none)
     * @param fileuri Optional URI pointing to detailed feedback JSON (empty string for none)
     * @param filehash KECCAK-256 hash of file content (bytes32(0) if no file or IPFS URI)
     * @param feedbackAuth Signed authorization from agent owner/operator (REQUIRED)
     * @notice feedbackAuth is a tuple (agentId, clientAddress, indexLimit, expiry, chainId, identityRegistry, signerAddress)
     *         signed using EIP-191 or ERC-1271
     */
    function giveFeedback(
        uint256 agentId,
        uint8 score,
        bytes32 tag1,
        bytes32 tag2,
        string calldata fileuri,
        bytes32 calldata filehash,
        bytes memory feedbackAuth
    ) external;
    
    /**
     * @dev Gets the reputation summary for an agent
     * @param agentId The agent token ID to query
     * @param clientAddresses Optional array of client addresses to filter feedback
     * @param tag1 Optional first tag to filter feedback (use bytes32(0) for no filter)
     * @param tag2 Optional second tag to filter feedback (use bytes32(0) for no filter)
     * @return feedbackCount The number of feedback entries matching the criteria
     * @return averageScore The average reputation score (0-100)
     * @notice If feedbackCount is 0, averageScore will be 0
     * @notice Scores range from 0 (worst) to 100 (best)
     */
    function getSummary(
        uint256 agentId,
        address[] calldata clientAddresses,
        bytes32 tag1,
        bytes32 tag2
    ) external view returns (uint256 feedbackCount, uint256 averageScore);
    
    /**
     * @dev Get the identity registry address
     * @return identityRegistry The address of the identity registry
     */
    function getIdentityRegistry() external view returns (address identityRegistry);
}
