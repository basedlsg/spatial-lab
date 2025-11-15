"""
Robot Communication System for Multi-Agent Coordination
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages robots can exchange."""
    STATUS_UPDATE = "status_update"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"


@dataclass
class RobotMessage:
    """Represents a message between robots."""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1


class RobotCommunicationSystem:
    """Communication system for robot coordination."""
    
    def __init__(self, max_range: float = 100.0):
        self.max_range = max_range
        self.message_queue: List[RobotMessage] = []
        self.robot_positions: Dict[str, tuple] = {}
        
    def register_robot(self, robot_id: str, position: tuple) -> None:
        """Register a robot in the communication system."""
        self.robot_positions[robot_id] = position
        
    def send_message(self, message: RobotMessage) -> bool:
        """Send a message between robots."""
        # Check if robots are in communication range
        if self._in_range(message.sender_id, message.receiver_id):
            self.message_queue.append(message)
            return True
        return False
        
    def receive_messages(self, robot_id: str) -> List[RobotMessage]:
        """Get messages for a specific robot."""
        messages = [msg for msg in self.message_queue if msg.receiver_id == robot_id]
        # Remove received messages from queue
        self.message_queue = [msg for msg in self.message_queue if msg.receiver_id != robot_id]
        return messages
        
    def _in_range(self, sender_id: str, receiver_id: str) -> bool:
        """Check if two robots are within communication range."""
        if sender_id not in self.robot_positions or receiver_id not in self.robot_positions:
            return False
            
        pos1 = self.robot_positions[sender_id]
        pos2 = self.robot_positions[receiver_id]
        distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        return distance <= self.max_range
        
    def update_robot_position(self, robot_id: str, position: tuple) -> None:
        """Update robot position for communication range calculations."""
        self.robot_positions[robot_id] = position
