// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC721Receiver} from "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import {IERC721} from "@openzeppelin/contracts/token/ERC721/IERC721.sol";

/**
 * @title AgentProxy
 * @notice A simple proxy contract that can register as an agent on ERC-8004 IdentityRegistry
 * @dev Used for testing since EOAs cannot receive NFTs via safeMint
 * 
 * The ERC-8004 IdentityRegistry uses _safeMint internally, which requires the receiver
 * to implement IERC721Receiver. This proxy contract fulfills that requirement.
 */
contract AgentProxy is IERC721Receiver {
    address public immutable owner;
    address public immutable identityRegistry;
    uint256 public agentId;
    
    event AgentRegistered(uint256 indexed agentId);
    
    constructor(address _identityRegistry) {
        owner = msg.sender;
        identityRegistry = _identityRegistry;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "AgentProxy: not owner");
        _;
    }
    
    /**
     * @notice Register this proxy as an agent
     * @return The assigned agent ID
     */
    function register() external onlyOwner returns (uint256) {
        (bool success, bytes memory data) = identityRegistry.call(
            abi.encodeWithSignature("register()")
        );
        require(success, "AgentProxy: registration failed");
        
        if (data.length >= 32) {
            agentId = abi.decode(data, (uint256));
        } else {
            // If register() doesn't return the ID, get it from balanceOf
            agentId = IERC721(identityRegistry).balanceOf(address(this));
        }
        
        emit AgentRegistered(agentId);
        return agentId;
    }
    
    /**
     * @notice Execute arbitrary call to another contract (for interacting with ReverseAuction, etc.)
     * @param target The target contract address
     * @param data The calldata to send
     * @return result The return data from the call
     */
    function execute(address target, bytes calldata data) external onlyOwner returns (bytes memory result) {
        bool success;
        (success, result) = target.call(data);
        require(success, "AgentProxy: execution failed");
    }
    
    /**
     * @notice Execute call with ETH value
     */
    function executeWithValue(address target, bytes calldata data) external payable onlyOwner returns (bytes memory result) {
        bool success;
        (success, result) = target.call{value: msg.value}(data);
        require(success, "AgentProxy: execution failed");
    }
    
    /**
     * @notice Approve another address to spend tokens on behalf of this proxy
     */
    function approveERC20(address token, address spender, uint256 amount) external onlyOwner {
        (bool success,) = token.call(
            abi.encodeWithSignature("approve(address,uint256)", spender, amount)
        );
        require(success, "AgentProxy: approval failed");
    }
    
    /**
     * @notice Required by IERC721Receiver to accept NFTs via safeMint/safeTransfer
     */
    function onERC721Received(
        address,
        address,
        uint256 tokenId,
        bytes calldata
    ) external override returns (bytes4) {
        // Store the agent ID when receiving from the identity registry
        if (msg.sender == identityRegistry) {
            agentId = tokenId;
        }
        return IERC721Receiver.onERC721Received.selector;
    }
    
    /**
     * @notice Receive ETH
     */
    receive() external payable {}
}
