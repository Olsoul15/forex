"""
Agent Communication Module for the AI Forex Trading System.

This module provides mechanisms for agents to communicate, share information,
and coordinate activities through message passing, shared memory, and events.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio
from queue import Queue
import threading
import time

from .base import BaseAgent
from forex_ai.utils.logging import get_logger
from forex_ai.exceptions import AgentCommunicationError

logger = get_logger(__name__)


@dataclass
class AgentMessage:
    """Data class for messages exchanged between agents."""

    sender_id: str
    content: Any
    message_type: str
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "sender_id": self.sender_id,
            "content": self.content,
            "message_type": self.message_type,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        # Convert timestamp from string to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)


class MessageBus:
    """
    Message bus for communication between agents.

    This class provides a centralized message passing facility for
    asynchronous communication between agents.
    """

    def __init__(self):
        """Initialize the message bus."""
        self.queues = {}  # Agent ID -> Queue
        self.handlers = {}  # Agent ID -> Dict[message_type, handler]
        self.subscribers = {}  # Topic -> List[Agent ID]
        self.lock = threading.RLock()

    def register_agent(self, agent_id: str) -> None:
        """
        Register an agent with the message bus.

        Args:
            agent_id: ID of the agent to register
        """
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = Queue()
                self.handlers[agent_id] = {}
                logger.info(f"Agent {agent_id} registered with message bus")

    def deregister_agent(self, agent_id: str) -> None:
        """
        Deregister an agent from the message bus.

        Args:
            agent_id: ID of the agent to deregister
        """
        with self.lock:
            if agent_id in self.queues:
                del self.queues[agent_id]

            if agent_id in self.handlers:
                del self.handlers[agent_id]

            # Remove from all subscribers lists
            for topic, subscribers in self.subscribers.items():
                if agent_id in subscribers:
                    subscribers.remove(agent_id)

            logger.info(f"Agent {agent_id} deregistered from message bus")

    def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the message bus.

        Args:
            message: Message to publish
        """
        with self.lock:
            if message.sender_id not in self.queues:
                logger.warning(f"Agent {message.sender_id} not found in message bus")
                return

            self.queues[message.sender_id].put(message)
            logger.info(f"Message published to agent {message.sender_id}")

    def subscribe(self, topic: str, agent_id: str) -> None:
        """
        Subscribe an agent to a topic.

        Args:
            topic: Topic to subscribe to
            agent_id: ID of the agent to subscribe
        """
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []

            if agent_id not in self.subscribers[topic]:
                self.subscribers[topic].append(agent_id)

            logger.info(f"Agent {agent_id} subscribed to topic {topic}")

    def unsubscribe(self, topic: str, agent_id: str) -> None:
        """
        Unsubscribe an agent from a topic.

        Args:
            topic: Topic to unsubscribe from
            agent_id: ID of the agent to unsubscribe
        """
        with self.lock:
            if topic in self.subscribers:
                if agent_id in self.subscribers[topic]:
                    self.subscribers[topic].remove(agent_id)

            logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")

    def handle(
        self, message_type: str, handler: Callable[[AgentMessage], None]
    ) -> None:
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Handler function to register
        """
        with self.lock:
            if message_type not in self.handlers:
                self.handlers[message_type] = []

            self.handlers[message_type].append(handler)
            logger.info(f"Handler registered for message type {message_type}")

    def unhandle(
        self, message_type: str, handler: Callable[[AgentMessage], None]
    ) -> None:
        """
        Unregister a handler for a specific message type.

        Args:
            message_type: Type of message to unhandle
            handler: Handler function to unregister
        """
        with self.lock:
            if message_type in self.handlers:
                if handler in self.handlers[message_type]:
                    self.handlers[message_type].remove(handler)

            logger.info(f"Handler unregistered for message type {message_type}")

    async def consume(self, agent_id: str) -> None:
        """
        Consume messages for an agent.

        Args:
            agent_id: ID of the agent to consume messages for
        """
        with self.lock:
            if agent_id not in self.queues:
                logger.warning(f"Agent {agent_id} not found in message bus")
                return

            queue = self.queues[agent_id]

            while True:
                message = queue.get()
                logger.info(f"Message received by agent {agent_id}")

                if message.sender_id not in self.handlers:
                    logger.warning(
                        f"No handlers registered for message type {message.message_type}"
                    )
                    continue

                for handler in self.handlers[message.sender_id]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Error handling message: {str(e)}")

    def register_handler(
        self, agent_id: str, message_type: str, handler: Callable
    ) -> None:
        """
        Register a message handler for an agent.

        Args:
            agent_id: ID of the agent
            message_type: Type of message to handle
            handler: Function to call when message arrives
        """
        with self.lock:
            if agent_id not in self.handlers:
                self.register_agent(agent_id)

            self.handlers[agent_id][message_type] = handler
            logger.debug(f"Agent {agent_id} registered handler for {message_type}")

    def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to a specific agent.

        Args:
            message: Message to send

        Raises:
            AgentCommunicationError: If the recipient agent is not registered
        """
        recipient_id = message.context.get("recipient_id")
        if not recipient_id:
            raise AgentCommunicationError("Message does not specify a recipient")

        with self.lock:
            if recipient_id not in self.queues:
                raise AgentCommunicationError(
                    f"Agent {recipient_id} not registered with message bus"
                )

            # Add to recipient's queue
            self.queues[recipient_id].put(message)
            logger.debug(f"Message {message.message_id} sent to agent {recipient_id}")

    def publish_message(self, message: AgentMessage, topic: str) -> None:
        """
        Publish a message to a topic.

        Args:
            message: Message to publish
            topic: Topic to publish to
        """
        with self.lock:
            if topic not in self.subscribers or not self.subscribers[topic]:
                logger.debug(f"No subscribers for topic {topic}")
                return

            # Add topic to message context
            message.context["topic"] = topic

            # Add to all subscribers' queues
            for agent_id in self.subscribers[topic]:
                if agent_id in self.queues:
                    self.queues[agent_id].put(message)

            logger.debug(
                f"Message {message.message_id} published to topic {topic} with {len(self.subscribers[topic])} subscribers"
            )

    def receive_message(
        self, agent_id: str, block: bool = False, timeout: float = None
    ) -> Optional[AgentMessage]:
        """
        Receive a message for an agent.

        Args:
            agent_id: ID of the agent
            block: Whether to block until a message is available
            timeout: Timeout for blocking (None = wait forever)

        Returns:
            AgentMessage if available, None otherwise

        Raises:
            AgentCommunicationError: If the agent is not registered
        """
        with self.lock:
            if agent_id not in self.queues:
                raise AgentCommunicationError(
                    f"Agent {agent_id} not registered with message bus"
                )

        # Get from queue without lock to avoid deadlock
        try:
            message = self.queues[agent_id].get(block=block, timeout=timeout)

            # Check if there's a handler for this message type
            if (
                agent_id in self.handlers
                and message.message_type in self.handlers[agent_id]
            ):
                # Call handler
                try:
                    self.handlers[agent_id][message.message_type](message)
                except Exception as e:
                    logger.error(
                        f"Error in message handler for agent {agent_id}: {str(e)}"
                    )

            return message
        except Exception:
            return None

    def process_messages(self, agent_id: str, max_messages: int = 10) -> int:
        """
        Process available messages for an agent.

        Args:
            agent_id: ID of the agent
            max_messages: Maximum number of messages to process

        Returns:
            Number of messages processed

        Raises:
            AgentCommunicationError: If the agent is not registered
        """
        with self.lock:
            if agent_id not in self.queues:
                raise AgentCommunicationError(
                    f"Agent {agent_id} not registered with message bus"
                )

        # Process messages without lock to avoid deadlock
        count = 0
        for _ in range(max_messages):
            message = self.receive_message(agent_id, block=False)
            if message is None:
                break
            count += 1

        return count


class SharedMemory:
    """
    Shared memory for inter-agent communication.

    This class provides a shared state that can be accessed by multiple agents.
    It ensures thread-safe access to the shared data.
    """

    def __init__(self):
        """Initialize the shared memory."""
        self.data = {}
        self.locks = {}
        self.global_lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from shared memory.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value associated with the key, or default if not found
        """
        with self.global_lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in shared memory.

        Args:
            key: Key to set
            value: Value to associate with the key
        """
        with self.global_lock:
            self.data[key] = value

    def update(self, key: str, value: Any) -> bool:
        """
        Update a value in shared memory if it exists.

        Args:
            key: Key to update
            value: New value

        Returns:
            True if updated, False if key not found
        """
        with self.global_lock:
            if key in self.data:
                self.data[key] = value
                return True
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from shared memory.

        Args:
            key: Key to delete

        Returns:
            True if deleted, False if key not found
        """
        with self.global_lock:
            if key in self.data:
                del self.data[key]
                return True
            return False

    def acquire_lock(self, key: str) -> bool:
        """
        Acquire a lock for a specific key.

        Args:
            key: Key to lock

        Returns:
            True if lock acquired, False if already locked
        """
        with self.global_lock:
            if key not in self.locks:
                self.locks[key] = threading.RLock()

            # Try to acquire the lock
            if self.locks[key].acquire(blocking=False):
                return True
            return False

    def release_lock(self, key: str) -> bool:
        """
        Release a lock for a specific key.

        Args:
            key: Key to unlock

        Returns:
            True if lock released, False if key not found or not locked
        """
        with self.global_lock:
            if key in self.locks:
                try:
                    self.locks[key].release()
                    return True
                except RuntimeError:
                    # Lock not owned by this thread
                    return False
            return False

    def with_lock(self, key: str):
        """
        Context manager for using a key with lock.

        Args:
            key: Key to lock

        Returns:
            Context manager
        """

        class LockContext:
            def __init__(self, shared_memory, key):
                self.shared_memory = shared_memory
                self.key = key

            def __enter__(self):
                with self.shared_memory.global_lock:
                    if self.key not in self.shared_memory.locks:
                        self.shared_memory.locks[self.key] = threading.RLock()
                    self.shared_memory.locks[self.key].acquire()
                return self.shared_memory.data.get(self.key)

            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    self.shared_memory.locks[self.key].release()
                except (KeyError, RuntimeError):
                    pass

        return LockContext(self, key)


class EventSystem:
    """
    Event system for pub/sub communication between agents.

    This class provides an event-based communication mechanism that allows
    agents to subscribe to and publish events without direct coupling.
    """

    def __init__(self):
        """Initialize the event system."""
        self.subscribers = {}  # Event type -> List of callbacks
        self.lock = threading.RLock()

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []

            if callback not in self.subscribers[event_type]:
                self.subscribers[event_type].append(callback)
                logger.debug(f"Added subscriber to event {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove
        """
        with self.lock:
            if (
                event_type in self.subscribers
                and callback in self.subscribers[event_type]
            ):
                self.subscribers[event_type].remove(callback)
                logger.debug(f"Removed subscriber from event {event_type}")

    def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event.

        Args:
            event_type: Type of event to publish
            data: Event data
        """
        callbacks = []

        # Get callbacks without holding the lock during notification
        with self.lock:
            if event_type in self.subscribers:
                callbacks = list(self.subscribers[event_type])

        # Notify subscribers
        for callback in callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}")

        logger.debug(f"Published event {event_type} to {len(callbacks)} subscribers")


class AgentCoordinator:
    """
    Coordinator for managing and orchestrating agent interactions.

    This class provides facilities for coordinating activities across multiple agents,
    including message routing, shared memory management, and event handling.
    """

    def __init__(self):
        """Initialize the agent coordinator."""
        self.agents = {}  # Agent ID -> Agent instance
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemory()
        self.event_system = EventSystem()
        self.lock = threading.RLock()

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the coordinator.

        Args:
            agent: Agent to register
        """
        with self.lock:
            if agent.id in self.agents:
                logger.warning(f"Agent {agent.id} already registered")
                return

            self.agents[agent.id] = agent
            self.message_bus.register_agent(agent.id)
            logger.info(f"Agent {agent.name} ({agent.id}) registered with coordinator")

    def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the coordinator.

        Args:
            agent_id: ID of the agent to deregister

        Returns:
            True if deregistered, False if not found
        """
        with self.lock:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not registered")
                return False

            self.message_bus.deregister_agent(agent_id)
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} deregistered from coordinator")
            return True

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent instance if found, None otherwise
        """
        with self.lock:
            return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: type) -> List[BaseAgent]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agents to find

        Returns:
            List of matching agent instances
        """
        with self.lock:
            return [
                agent for agent in self.agents.values() if isinstance(agent, agent_type)
            ]

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        message_type: str,
        priority: int = 0,
    ) -> str:
        """
        Send a message from one agent to another.

        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            content: Message content
            message_type: Type of message
            priority: Message priority

        Returns:
            Message ID

        Raises:
            AgentCommunicationError: If sender or recipient is not registered
        """
        with self.lock:
            if sender_id not in self.agents:
                raise AgentCommunicationError(
                    f"Sender agent {sender_id} not registered"
                )

            if recipient_id not in self.agents:
                raise AgentCommunicationError(
                    f"Recipient agent {recipient_id} not registered"
                )

        # Create message
        message = AgentMessage(
            sender_id=sender_id,
            content=content,
            message_type=message_type,
            priority=priority,
            context={"recipient_id": recipient_id},
        )

        # Send message
        self.message_bus.send_message(message)

        return message.message_id

    def broadcast_message(
        self, sender_id: str, content: Any, message_type: str, priority: int = 0
    ) -> str:
        """
        Broadcast a message to all registered agents.

        Args:
            sender_id: ID of the sending agent
            content: Message content
            message_type: Type of message
            priority: Message priority

        Returns:
            Message ID

        Raises:
            AgentCommunicationError: If sender is not registered
        """
        with self.lock:
            if sender_id not in self.agents:
                raise AgentCommunicationError(
                    f"Sender agent {sender_id} not registered"
                )

            recipient_ids = list(self.agents.keys())

        # Send to all agents except sender
        message_id = None
        for recipient_id in recipient_ids:
            if recipient_id != sender_id:
                message_id = self.send_message(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    content=content,
                    message_type=message_type,
                    priority=priority,
                )

        return message_id

    def publish_event(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event to publish
            data: Event data
        """
        self.event_system.publish(event_type, data)

    def set_shared_data(self, key: str, value: Any) -> None:
        """
        Set a value in shared memory.

        Args:
            key: Key to set
            value: Value to associate with the key
        """
        self.shared_memory.set(key, value)

    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """
        Get a value from shared memory.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value associated with the key, or default if not found
        """
        return self.shared_memory.get(key, default)

    def process_agent_messages(self, agent_id: str, max_messages: int = 10) -> int:
        """
        Process messages for a specific agent.

        Args:
            agent_id: ID of the agent
            max_messages: Maximum number of messages to process

        Returns:
            Number of messages processed

        Raises:
            AgentCommunicationError: If agent is not registered
        """
        with self.lock:
            if agent_id not in self.agents:
                raise AgentCommunicationError(f"Agent {agent_id} not registered")

        return self.message_bus.process_messages(agent_id, max_messages)

    def process_all_messages(self, max_per_agent: int = 10) -> Dict[str, int]:
        """
        Process messages for all registered agents.

        Args:
            max_per_agent: Maximum number of messages to process per agent

        Returns:
            Dictionary mapping agent IDs to number of messages processed
        """
        results = {}

        with self.lock:
            agent_ids = list(self.agents.keys())

        for agent_id in agent_ids:
            try:
                count = self.process_agent_messages(agent_id, max_per_agent)
                results[agent_id] = count
            except Exception as e:
                logger.error(
                    f"Error processing messages for agent {agent_id}: {str(e)}"
                )
                results[agent_id] = 0

        return results
