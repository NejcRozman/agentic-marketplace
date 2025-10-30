// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC721} from "@openzeppelin/contracts/token/ERC721/IERC721.sol";

/**
 * @title IIdentityRegistry
 * @dev Interface for ERC-8004 Identity Registry
 * @notice This is the core registry that manages agent identities as ERC-721 NFTs
 */
interface IIdentityRegistry is IERC721 {
    /**
     * @dev Returns the owner of the agent NFT
     * @param agentId The agent token ID
     * @return owner The address that owns this agent
     * @notice Inherited from IERC721, but explicitly documented for clarity
     */
    function ownerOf(uint256 agentId) external view returns (address owner);
    
    /**
     * @dev Checks if an agent exists in the registry
     * @param agentId The agent token ID to check
     * @return exists True if the agent has been minted
     * @notice Can be implemented by checking if ownerOf doesn't revert or returns non-zero
     */
    function exists(uint256 agentId) external view returns (bool exists);
}
