// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IReputationRegistry
 * @dev Interface for ERC-8004 Reputation Registry
 * @notice Manages reputation feedback for agents with scores from 0-100
 */
interface IReputationRegistry {
    /**
     * @dev Gets the reputation summary for an agent
     * @param agentId The agent token ID to query
     * @param clientAddresses Optional array of client addresses to filter feedback
     * @param tag1 Optional first tag to filter feedback (use 0 for no filter)
     * @param tag2 Optional second tag to filter feedback (use 0 for no filter)
     * @return feedbackCount The number of feedback entries matching the criteria
     * @return averageScore The average reputation score (0-100)
     * @notice If feedbackCount is 0, averageScore will be 0
     * @notice Scores range from 0 (worst) to 100 (best)
     */
    function getSummary(
        uint256 agentId,
        address[] calldata clientAddresses,
        uint256 tag1,
        uint256 tag2
    ) external view returns (uint256 feedbackCount, uint256 averageScore);
}
