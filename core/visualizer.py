import folium
import os
import random
import requests
import json
import polyline
from typing import List, Dict
from core.data_structures import RvrpState, ProblemData
from config import PathConfig

class RouteVisualizer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir if output_dir else PathConfig.RESULTS_DIR
        self.cache_dir = os.path.join("inputs", "DistTimeMatrix")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.geometry_cache = {}
        self.current_depot_id = None
        
        self.color_map = {
            'MC': 'green',          
            'AUV': 'orange',        
            '4w': 'blue',           
            '6w': 'purple',         
            '10w': 'darkpurple',        
            '40ft': 'black'         
        }
        self.fallback_colors = ['cadetblue', 'darkgreen', 'darkblue', 'gray']

    def _get_color(self, vehicle_name: str) -> str:
        for key, color in self.color_map.items():
            if key in vehicle_name:
                return color
        return random.choice(self.fallback_colors)

    def _load_cache(self, depot_id: str):
        """Load cache file specific to a Depot"""
        self.current_depot_id = depot_id
        cache_path = os.path.join(self.cache_dir, f"geometry_{depot_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    self.geometry_cache = json.load(f)
            except Exception as e:
                print(f"  [Visualizer] Error loading cache: {e}")
                self.geometry_cache = {}
        else:
            self.geometry_cache = {}

    def _save_cache(self):
        """Persist cache to disk"""
        if self.current_depot_id:
            cache_path = os.path.join(self.cache_dir, f"geometry_{self.current_depot_id}.json")
            try:
                with open(cache_path, 'w') as f:
                    json.dump(self.geometry_cache, f)
            except Exception as e:
                print(f"  [Visualizer] Error saving cache: {e}")

    def _generate_key(self, points: List[tuple]) -> str:
        """Create a unique string key for a sequence of points"""
        return ";".join([f"{p[0]:.5f},{p[1]:.5f}" for p in points])

    def _get_osrm_geometry(self, points: List[tuple]) -> List[tuple]:
        """
        Get geometry from Cache first, then OSRM.
        """
        if len(points) < 2: return points
        
        # 1. Check Cache
        key = self._generate_key(points)
        if key in self.geometry_cache:
            encoded = self.geometry_cache[key]
            try:
                return polyline.decode(encoded)
            except:
                pass
        
        # 2. Call OSRM API
        # OSRM expects: lon,lat;lon,lat
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in points])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=polyline"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok' and data['routes']:
                    encoded = data['routes'][0]['geometry']
                    
                    # 3. Update Cache
                    self.geometry_cache[key] = encoded
                    
                    return polyline.decode(encoded)
        except Exception as e:
            print(f"  [Visualizer] OSRM Request Failed: {e}")
        
        return points # Fallback to straight lines

    def visualize_solution(self, solution: RvrpState, data: ProblemData, filename: str = "optimized_route.html"):
        """
        Generates map with persistent geometry caching.
        """
        if not hasattr(data, 'coords') or len(data.coords) == 0:
            return

        # 1. INIT CACHE FOR THIS DEPOT
        depot_id = str(data.node_ids[0])
        self._load_cache(depot_id)

        depot_coords = data.coords[0]
        m = folium.Map(location=[depot_coords[0], depot_coords[1]], zoom_start=12, tiles='CartoDB positron')

        # Depot Marker
        folium.Marker(
            location=[depot_coords[0], depot_coords[1]],
            popup=f"<b>DEPOT: {depot_id}</b>",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)

        # 2. DRAW ROUTES
        for i, route in enumerate(solution.routes):
            v_name = route.vehicle_type.name
            color = self._get_color(v_name)
            
            # Prepare Key Points
            path_indices = [0] + route.node_sequence + [0]
            path_coords = [tuple(data.coords[idx]) for idx in path_indices]
            
            # Get Geometry (Cached or Live)
            real_path_coords = self._get_osrm_geometry(path_coords)
            
            layer_name = f"R{i+1}: {v_name} ({len(route.node_sequence)} stops)"
            fg = folium.FeatureGroup(name=layer_name)
            
            # Polyline
            folium.PolyLine(
                locations=real_path_coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"Route {i+1} ({v_name})"
            ).add_to(fg)
            
            # Utilization & Summary
            util_percent = route.capacity_utilization * 100
            warning_html = "<br><b style='color:red;'>⚠️ LOW UTIL</b>" if util_percent < 50.0 else ""
            
            # Popup logic
            mid_idx = len(real_path_coords) // 2
            folium.Marker(
                real_path_coords[mid_idx], 
                icon=folium.DivIcon(html=f"""<div style="font-size:0pt">.</div>"""),
                popup=folium.Popup(
                    f"<b>Route {i+1}</b><br>Vehicle: {v_name}<br>Util: {util_percent:.1f}% {warning_html}<br>Cost: {route.cost:,.0f}", 
                    max_width=200
                )
            ).add_to(fg)
            
            # Customer Markers
            for seq_idx, node_idx in enumerate(route.node_sequence):
                demand_str = f"{data.demands_kg[node_idx]:.0f}kg"
                tw_start, tw_end = data.time_windows[node_idx]
                tw_str = f"{int(tw_start)//60:02d}:{int(tw_start)%60:02d}-{int(tw_end)//60:02d}:{int(tw_end)%60:02d}"
                
                folium.CircleMarker(
                    location=[data.coords[node_idx][0], data.coords[node_idx][1]],
                    radius=4, color=color, fill=True, fill_color='white', fill_opacity=1.0,
                    popup=folium.Popup(f"<b>{data.node_ids[node_idx]}</b><br>Seq: {seq_idx+1}<br>Dem: {demand_str}<br>TW: {tw_str}", max_width=200)
                ).add_to(fg)

            fg.add_to(m)

        # Unassigned
        if solution.unassigned:
            fg_un = folium.FeatureGroup(name="Unassigned", show=True)
            for u_idx in solution.unassigned:
                folium.CircleMarker(
                    location=[data.coords[u_idx][0], data.coords[u_idx][1]],
                    radius=6, color='red', fill=True, fill_color='darkred',
                    popup=f"Unassigned: {data.node_ids[u_idx]}"
                ).add_to(fg_un)
            fg_un.add_to(m)

        folium.LayerControl().add_to(m)
        
        # 3. SAVE OUTPUT & CACHE
        save_path = os.path.join(self.output_dir, filename)
        m.save(save_path)
        self._save_cache() # Persist cache
        print(f"  > [Visualizer] Map saved: {save_path}")