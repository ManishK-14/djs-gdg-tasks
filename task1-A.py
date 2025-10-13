import numpy as np

forest_map = np.array([
[1,0,0,0,1],
[1,0,1,1,1],
[1,1,0,1,1],
[1,0,1,1,0],
[0,1,0,1,1]
])
print(forest_map.shape)
m = int(input("Enter the value of m: "))
#for r and c for centre

r = int(input("Enter the value of r: "))
c = int(input("Enter the value of c: "))
print(f"(r,c) = ({r},{c})")

extraction_zone = np.zeros((m,m))

print(forest_map[r][c])

up_down = m//3 #floor division 
# i will check how much i need from centre for a given m 
extraction_zone = forest_map[r-up_down : r+up_down+1, c-up_down: c+up_down+1]

print(extraction_zone)
Lal_Chandan_trees =0
Lal_Chandan_trees = np.sum(extraction_zone[extraction_zone==1])

print(Lal_Chandan_trees)