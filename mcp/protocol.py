from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid

class MCPMessage(BaseModel):
    """
    定义了组件之间通信的消息格式 (Message Format)。
    """
    sender_id: str
    receiver_id: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    data: Dict[str, Any]
    thought: Optional[str] = None