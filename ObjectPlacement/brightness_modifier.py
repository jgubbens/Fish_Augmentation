import math

class BrightnessModifier:

    # Accepts coords in tuples {x, y}, where they are scaled from 0 to 1
    def __init__(self, source_coords, adj_intensity):
        self.source_coords = source_coords
        self.adj_intensity = adj_intensity
    
    def _find_multiplier(self, coords):
        distance = math.sqrt((coords[0] - self.source_coords[0]) ** 2 +
                             (coords[1] - self.source_coords[1]) ** 2)
        return 1 - (distance * self.adj_intensity)
    
    # Accepts coords in tuples {x, y}, where they are scaled from 0 to 1
    def find_brightness(self, org_coords, adj_coords):
        org_multiplier = self._find_multiplier(org_coords)
        adj_multiplier = self._find_multiplier(adj_coords)
        return adj_multiplier / org_multiplier
'''
# Function to print a brightness map
def print_brightness_map(source_coords, max_adj, grid_size=11):
    bm = BrightnessModifier(source_coords, max_adj)
    
    print(f"Brightness Map (Source: {source_coords}, Max Adjustment: {max_adj})\n")
    for y in range(grid_size):
        for x in range(grid_size):
            # Scale grid coordinates from 0 to 1
            coords = (x / (grid_size - 1), y / (grid_size - 1))
            brightness = bm.find_brightness(source_coords, coords)
            print(f"{brightness:.2f}", end="\t")
        print()  # Newline for the next row

source_coords = (0.8, 0.2)  # Center of the grid
max_adj = 0.5  # Maximum adjustment
print_brightness_map(source_coords, max_adj)'''