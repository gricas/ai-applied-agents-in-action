from pydantic import BaseModel
from typing import Literal

class ClasificationResponse(BaseModel):
    category: Literal["Dogs", "Baseball", "Coffee"]
