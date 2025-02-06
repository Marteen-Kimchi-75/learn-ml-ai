# PRINTING
print("Python is simple to write.")

# VARIABLES
integer_var = 8
floating_point_var = 9.5
string_var = "Python"

list_var = [1, 4, 7, 15]

dictionary_var = {
    "Name" : "Machine Learning",
    "Application": "Any field"
}

# CONTROL FLOW
if integer_var > 5:
    print("This number is greater than 5")
else:
    print("This number is not greater than 5")

# LOOP
for i in range(10):
    print(i, end=" ")
print()

# FUNCTIONS
def sum(a, b):
    return a + b;
print(sum(1, 2))

# CLASSES AND OBJECTS
class Vehicle:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    def getName(self):
        return self.name

    def getPrice(self):
        return self.price

my_car = Vehicle("Maruti Car", 200000)
print(my_car.getName())
print(my_car.getPrice())