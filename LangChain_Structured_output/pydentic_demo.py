from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Cppandey" #default value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, le=10, default=7, description='A decimal value represting the value of the student')


new_student = {"age": '32', "email": "cp.o.pandey@gmail.com"}

student = Student(**new_student)

student_dict = dict(student)
student_json = student.model_dump_json()

print(student)
print(student_dict)
print(student_json)