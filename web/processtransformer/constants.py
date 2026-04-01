import enum

@enum.unique
class Task(enum.Enum):
  """Look up for tasks."""
  
  NEXT_ACTIVITY = "next_activity"
  NEXT_TIME = "next_time"
  REMAINING_TIME = "remaining_time"
  NEXT_MESSAGE_SEND = "next_message_send"
  NEXT_TIME_MESSAGE = "next_time_message"
  REMAINING_TIME_PARTICIPANT = "remaining_time_participant"


