// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./IERC20.sol";

/// @title A simple token vault.
contract Vault {
    error Unauthorized(address caller);

    event Deposited(address indexed from, uint256 amount);

    struct Account {
        uint256 balance;
        uint64 updated;
    }

    enum State { Open, Closed }

    address public immutable owner;
    mapping(address => Account) private accounts;
    State public state = State.Open;

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized(msg.sender);
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Deposits ether.
    function deposit() external payable {
        accounts[msg.sender].balance += msg.value;
        accounts[msg.sender].updated = uint64(block.timestamp);
        emit Deposited(msg.sender, msg.value);
    }

    function balanceOf(address who) external view returns (uint256) {
        return accounts[who].balance;
    }

    function close() external onlyOwner {
        state = State.Closed;
    }

    receive() external payable {}
}

interface IVault {
    function deposit() external payable;
}

library Math {
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }
}
