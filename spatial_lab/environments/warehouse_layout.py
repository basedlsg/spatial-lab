"""
Warehouse Layout Generator

Creates realistic warehouse layouts with shelves, aisles, loading docks,
and other infrastructure for spatial reasoning experiments.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class WarehouseShelf:
    """Represents a storage shelf in the warehouse"""
    shelf_id: str
    position: Tuple[float, float, float]  # (x, y, z)
    dimensions: Tuple[float, float, float]  # (width, depth, height)
    orientation: float  # rotation in radians
    shelf_type: str = "standard"  # standard, tall, wide, special
    items: List[str] = field(default_factory=list)
    max_capacity: int = 20
    
    def get_items(self) -> List[str]:
        """Get list of items on this shelf"""
        return self.items.copy()
    
    def add_item(self, item_id: str) -> bool:
        """Add item to shelf if space available"""
        if len(self.items) < self.max_capacity:
            self.items.append(item_id)
            return True
        return False
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from shelf"""
        if item_id in self.items:
            self.items.remove(item_id)
            return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "shelf_id": self.shelf_id,
            "position": self.position,
            "dimensions": self.dimensions,
            "orientation": self.orientation,
            "shelf_type": self.shelf_type,
            "items": self.items,
            "capacity": f"{len(self.items)}/{self.max_capacity}"
        }


@dataclass
class WarehouseZone:
    """Represents a functional zone in the warehouse"""
    zone_id: str
    zone_type: str  # storage, picking, packing, shipping, receiving
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x_min, y_min), (x_max, y_max))
    access_points: List[Tuple[float, float]] = field(default_factory=list)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is within this zone"""
        x, y = point
        (x_min, y_min), (x_max, y_max) = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of the zone"""
        (x_min, y_min), (x_max, y_max) = self.bounds
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)


@dataclass
class WarehouseAisle:
    """Represents an aisle between shelves"""
    aisle_id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    width: float
    aisle_type: str = "main"  # main, secondary, cross
    
    def get_waypoints(self, num_points: int = 5) -> List[Tuple[float, float]]:
        """Get waypoints along the aisle for navigation"""
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = self.start_point[0] + t * (self.end_point[0] - self.start_point[0])
            y = self.start_point[1] + t * (self.end_point[1] - self.start_point[1])
            waypoints.append((x, y))
        return waypoints


@dataclass
class WarehouseLayout:
    """Complete warehouse layout"""
    layout_id: str
    dimensions: Tuple[float, float]  # (width, height)
    shelves: List[WarehouseShelf] = field(default_factory=list)
    zones: List[WarehouseZone] = field(default_factory=list)
    aisles: List[WarehouseAisle] = field(default_factory=list)
    obstacles: List[Dict] = field(default_factory=list)  # Static obstacles
    
    def get_shelf_by_id(self, shelf_id: str) -> Optional[WarehouseShelf]:
        """Get shelf by ID"""
        for shelf in self.shelves:
            if shelf.shelf_id == shelf_id:
                return shelf
        return None
    
    def get_zone_by_type(self, zone_type: str) -> List[WarehouseZone]:
        """Get all zones of specified type"""
        return [zone for zone in self.zones if zone.zone_type == zone_type]
    
    def is_position_valid(self, position: Tuple[float, float], robot_radius: float = 0.5) -> bool:
        """Check if position is valid for robot navigation"""
        x, y = position
        
        # Check warehouse bounds
        if x < robot_radius or x > self.dimensions[0] - robot_radius:
            return False
        if y < robot_radius or y > self.dimensions[1] - robot_radius:
            return False
        
        # Check collision with shelves
        for shelf in self.shelves:
            shelf_x, shelf_y, _ = shelf.position
            shelf_w, shelf_d, _ = shelf.dimensions
            
            # Simple bounding box collision check
            if (shelf_x - shelf_w/2 - robot_radius <= x <= shelf_x + shelf_w/2 + robot_radius and
                shelf_y - shelf_d/2 - robot_radius <= y <= shelf_y + shelf_d/2 + robot_radius):
                return False
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            obs_x, obs_y = obstacle["position"][:2]
            obs_radius = obstacle.get("radius", 1.0)
            
            distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < obs_radius + robot_radius:
                return False
        
        return True
    
    def get_nearest_aisle(self, position: Tuple[float, float]) -> Optional[WarehouseAisle]:
        """Get the nearest aisle to a position"""
        min_distance = float('inf')
        nearest_aisle = None
        
        for aisle in self.aisles:
            # Calculate distance to aisle (simplified as distance to midpoint)
            mid_x = (aisle.start_point[0] + aisle.end_point[0]) / 2
            mid_y = (aisle.start_point[1] + aisle.end_point[1]) / 2
            
            distance = np.sqrt((position[0] - mid_x)**2 + (position[1] - mid_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_aisle = aisle
        
        return nearest_aisle
    
    def to_dict(self) -> Dict:
        """Convert layout to dictionary representation"""
        return {
            "layout_id": self.layout_id,
            "dimensions": self.dimensions,
            "shelves": [shelf.to_dict() for shelf in self.shelves],
            "zones": [
                {
                    "zone_id": zone.zone_id,
                    "zone_type": zone.zone_type,
                    "bounds": zone.bounds,
                    "center": zone.get_center()
                } for zone in self.zones
            ],
            "aisles": [
                {
                    "aisle_id": aisle.aisle_id,
                    "start": aisle.start_point,
                    "end": aisle.end_point,
                    "width": aisle.width,
                    "type": aisle.aisle_type
                } for aisle in self.aisles
            ],
            "obstacles": self.obstacles
        }


class WarehouseLayoutGenerator:
    """Generates realistic warehouse layouts"""
    
    def __init__(self, width: float = 50.0, height: float = 30.0, num_shelves: int = 20):
        self.width = width
        self.height = height
        self.num_shelves = num_shelves
        self.shelf_templates = self._create_shelf_templates()
        
    def _create_shelf_templates(self) -> Dict[str, Dict]:
        """Create templates for different shelf types"""
        return {
            "standard": {
                "dimensions": (2.0, 1.0, 3.0),
                "capacity": 20,
                "common": True
            },
            "tall": {
                "dimensions": (2.0, 1.0, 5.0),
                "capacity": 30,
                "common": False
            },
            "wide": {
                "dimensions": (4.0, 1.0, 3.0),
                "capacity": 40,
                "common": False
            },
            "compact": {
                "dimensions": (1.5, 0.8, 2.5),
                "capacity": 15,
                "common": True
            }
        }
    
    async def generate_layout(self, layout_type: str = "standard") -> WarehouseLayout:
        """Generate a complete warehouse layout"""
        
        layout_id = f"warehouse_{uuid.uuid4().hex[:8]}"
        
        if layout_type == "standard":
            return await self._generate_standard_layout(layout_id)
        elif layout_type == "complex":
            return await self._generate_complex_layout(layout_id)
        else:
            return await self._generate_standard_layout(layout_id)
    
    async def _generate_standard_layout(self, layout_id: str) -> WarehouseLayout:
        """Generate a standard warehouse layout with organized shelving"""
        
        layout = WarehouseLayout(
            layout_id=layout_id,
            dimensions=(self.width, self.height)
        )
        
        # Create zones
        await self._create_warehouse_zones(layout)
        
        # Generate shelves in organized rows
        await self._generate_organized_shelves(layout)
        
        # Create aisles between shelf rows
        await self._generate_aisles(layout)
        
        # Add some obstacles (pillars, equipment)
        await self._add_obstacles(layout)
        
        # Populate shelves with items
        await self._populate_shelves(layout)
        
        return layout
    
    async def _create_warehouse_zones(self, layout: WarehouseLayout):
        """Create functional zones in the warehouse"""
        
        # Receiving zone (left side)
        receiving_zone = WarehouseZone(
            zone_id="receiving_001",
            zone_type="receiving",
            bounds=((0, 0), (8, self.height)),
            access_points=[(4, 2), (4, self.height - 2)]
        )
        layout.zones.append(receiving_zone)
        
        # Storage zone (center)
        storage_zone = WarehouseZone(
            zone_id="storage_001", 
            zone_type="storage",
            bounds=((8, 0), (self.width - 8, self.height)),
            access_points=[]
        )
        layout.zones.append(storage_zone)
        
        # Shipping zone (right side)
        shipping_zone = WarehouseZone(
            zone_id="shipping_001",
            zone_type="shipping", 
            bounds=((self.width - 8, 0), (self.width, self.height)),
            access_points=[(self.width - 4, 2), (self.width - 4, self.height - 2)]
        )
        layout.zones.append(shipping_zone)
        
        # Picking zones (scattered throughout storage area)
        for i in range(3):
            picking_zone = WarehouseZone(
                zone_id=f"picking_{i:03d}",
                zone_type="picking",
                bounds=(
                    (10 + i * 8, 5),
                    (16 + i * 8, 15)
                ),
                access_points=[(12 + i * 8, 10)]
            )
            layout.zones.append(picking_zone)
    
    async def _generate_organized_shelves(self, layout: WarehouseLayout):
        """Generate shelves in organized rows"""
        
        storage_zone = layout.get_zone_by_type("storage")[0]
        (x_min, y_min), (x_max, y_max) = storage_zone.bounds
        
        # Calculate shelf placement
        shelf_width = 2.0
        shelf_depth = 1.0
        aisle_width = 3.0
        
        # Number of rows and shelves per row
        available_width = x_max - x_min - 2  # Leave margin
        row_spacing = shelf_depth * 2 + aisle_width
        num_rows = int((y_max - y_min - 2) / row_spacing)
        
        shelves_per_row = int(available_width / (shelf_width + 1))
        
        shelf_count = 0
        for row in range(num_rows):
            if shelf_count >= self.num_shelves:
                break
                
            y_pos = y_min + 2 + row * row_spacing + shelf_depth / 2
            
            for col in range(shelves_per_row):
                if shelf_count >= self.num_shelves:
                    break
                
                x_pos = x_min + 2 + col * (shelf_width + 1) + shelf_width / 2
                
                # Choose shelf type (mostly standard, some variety)
                shelf_type = "standard"
                if random.random() < 0.2:  # 20% chance of other types
                    shelf_type = random.choice(["tall", "wide", "compact"])
                
                template = self.shelf_templates[shelf_type]
                
                shelf = WarehouseShelf(
                    shelf_id=f"shelf_{shelf_count:03d}",
                    position=(x_pos, y_pos, template["dimensions"][2] / 2),
                    dimensions=template["dimensions"],
                    orientation=0.0,
                    shelf_type=shelf_type,
                    max_capacity=template["capacity"]
                )
                
                layout.shelves.append(shelf)
                shelf_count += 1
    
    async def _generate_aisles(self, layout: WarehouseLayout):
        """Generate aisles between shelf rows for navigation"""
        
        storage_zone = layout.get_zone_by_type("storage")[0]
        (x_min, y_min), (x_max, y_max) = storage_zone.bounds
        
        # Main horizontal aisles
        shelf_rows = {}
        for shelf in layout.shelves:
            row_y = int(shelf.position[1] / 5) * 5  # Group by approximate Y position
            if row_y not in shelf_rows:
                shelf_rows[row_y] = []
            shelf_rows[row_y].append(shelf)
        
        # Create aisles between rows
        aisle_count = 0
        sorted_rows = sorted(shelf_rows.keys())
        
        for i in range(len(sorted_rows) - 1):
            current_row_y = sorted_rows[i]
            next_row_y = sorted_rows[i + 1]
            
            aisle_y = (current_row_y + next_row_y) / 2
            
            aisle = WarehouseAisle(
                aisle_id=f"aisle_h_{aisle_count:03d}",
                start_point=(x_min + 1, aisle_y),
                end_point=(x_max - 1, aisle_y),
                width=3.0,
                aisle_type="main"
            )
            layout.aisles.append(aisle)
            aisle_count += 1
        
        # Vertical cross aisles
        for i in range(3):
            x_pos = x_min + 5 + i * ((x_max - x_min - 10) / 2)
            
            aisle = WarehouseAisle(
                aisle_id=f"aisle_v_{i:03d}",
                start_point=(x_pos, y_min + 1),
                end_point=(x_pos, y_max - 1),
                width=2.5,
                aisle_type="cross"
            )
            layout.aisles.append(aisle)
    
    async def _add_obstacles(self, layout: WarehouseLayout):
        """Add static obstacles like pillars and equipment"""
        
        # Add a few pillars for structural realism
        pillar_positions = [
            (15, 8, "pillar"),
            (25, 12, "pillar"),
            (35, 18, "pillar"),
            (20, 25, "equipment"),
            (30, 5, "equipment")
        ]
        
        for x, y, obs_type in pillar_positions:
            if x < self.width and y < self.height:
                obstacle = {
                    "obstacle_id": f"{obs_type}_{len(layout.obstacles):03d}",
                    "type": obs_type,
                    "position": (x, y, 1.0),
                    "radius": 0.8 if obs_type == "pillar" else 1.2
                }
                layout.obstacles.append(obstacle)
    
    async def _populate_shelves(self, layout: WarehouseLayout):
        """Populate shelves with items"""
        
        item_types = [
            "electronics", "clothing", "books", "tools", "food",
            "toys", "furniture", "sports", "automotive", "medical"
        ]
        
        item_count = 0
        
        for shelf in layout.shelves:
            # Fill shelf to 60-90% capacity
            target_items = int(shelf.max_capacity * random.uniform(0.6, 0.9))
            
            for _ in range(target_items):
                item_type = random.choice(item_types)
                item_id = f"{item_type}_{item_count:05d}"
                
                shelf.add_item(item_id)
                item_count += 1
    
    async def _generate_complex_layout(self, layout_id: str) -> WarehouseLayout:
        """Generate a more complex warehouse layout with irregular features"""
        
        # Start with standard layout
        layout = await self._generate_standard_layout(layout_id)
        
        # Add complexity: irregular shelf arrangements, more obstacles, etc.
        await self._add_irregular_features(layout)
        
        return layout
    
    async def _add_irregular_features(self, layout: WarehouseLayout):
        """Add irregular features to make layout more complex"""
        
        # Add some randomly rotated shelves
        for shelf in random.sample(layout.shelves, min(5, len(layout.shelves))):
            shelf.orientation = random.uniform(-np.pi/4, np.pi/4)
        
        # Add more obstacles in random positions
        for _ in range(8):
            while True:
                x = random.uniform(10, self.width - 10)
                y = random.uniform(5, self.height - 5)
                
                if layout.is_position_valid((x, y), robot_radius=2.0):
                    obstacle = {
                        "obstacle_id": f"random_{len(layout.obstacles):03d}",
                        "type": "equipment",
                        "position": (x, y, 1.0),
                        "radius": random.uniform(0.8, 1.5)
                    }
                    layout.obstacles.append(obstacle)
                    break
    
    def get_safe_robot_positions(self, layout: WarehouseLayout, num_positions: int) -> List[Tuple[float, float]]:
        """Generate safe starting positions for robots"""
        
        safe_positions = []
        max_attempts = num_positions * 10
        attempts = 0
        
        while len(safe_positions) < num_positions and attempts < max_attempts:
            # Try to place robots in aisles or open areas
            x = random.uniform(2, layout.dimensions[0] - 2)
            y = random.uniform(2, layout.dimensions[1] - 2)
            
            if layout.is_position_valid((x, y), robot_radius=0.5):
                # Check minimum distance from other robots
                min_distance = 3.0
                too_close = False
                
                for existing_pos in safe_positions:
                    distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    safe_positions.append((x, y))
            
            attempts += 1
        
        return safe_positions 