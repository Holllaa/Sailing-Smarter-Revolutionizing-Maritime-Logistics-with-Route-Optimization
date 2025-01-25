import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from datetime import datetime, timedelta
import haversine as hs
from typing import Dict, List, Tuple, Optional

class MaritimeRouteOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.route_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.port_graph = nx.Graph()
        self.transshipment_hubs = set()
        self.land_routes = nx.Graph()

    def haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return hs.haversine(point1, point2, unit=hs.Unit.KILOMETERS)

    def prepare_training_data(self, historical_routes: pd.DataFrame) -> None:
        features = ['distance', 'cargo_weight', 'temperature_requirements', 'fuel_cost', 'weather_condition', 'port_congestion', 'season', 'cargo_type']
        X = historical_routes[features]
        y = historical_routes['efficiency_score']
        X_scaled = self.scaler.fit_transform(X)
        self.route_model.fit(X_scaled, y)

    def predict_route_efficiency(self, route_features: Dict) -> float:
        features_scaled = self.scaler.transform([list(route_features.values())])
        return self.route_model.predict(features_scaled)[0]

    def find_transshipment_options(self, origin: Tuple[float, float], destination: Tuple[float, float], cargo_details: Dict) -> List[Dict]:
        options = []
        for hub in self.transshipment_hubs:
            dist_to_hub = self.haversine_distance(origin, hub)
            dist_from_hub = self.haversine_distance(hub, destination)
            total_dist = dist_to_hub + dist_from_hub
            if self.is_hub_suitable(hub, cargo_details):
                options.append({'hub': hub, 'total_distance': total_dist, 'main_route_distance': dist_to_hub, 'feeder_distance': dist_from_hub})
        return sorted(options, key=lambda x: x['total_distance'])

    def calculate_fuel_consumption(self, distance: float, cargo_weight: float, ship_specs: Dict) -> float:
        base_consumption = ship_specs['base_fuel_rate']
        cargo_factor = 1 + (cargo_weight / ship_specs['max_capacity']) * 0.3
        return base_consumption * distance * cargo_factor

    def check_temperature_requirements(self, route: List[Tuple[float, float]], required_temp: float) -> bool:
        return True

    def find_optimal_route(self, origin: Tuple[float, float], destination: Tuple[float, float], cargo_details: Dict, ship_specs: Dict) -> Dict:
        direct_route = [origin, destination]
        direct_score = self.predict_route_efficiency({
            'distance': self.haversine_distance(origin, destination),
            'cargo_weight': cargo_details['weight'],
            'temperature_requirements': cargo_details.get('required_temperature', 0),
            'fuel_cost': 1000,
            'weather_condition': 1,
            'port_congestion': 0.5,
            'season': 1,
            'cargo_type': 1
        })

        transshipment_options = self.find_transshipment_options(origin, destination, cargo_details)
        best_transshipment = None
        best_transshipment_score = float('-inf')

        for option in transshipment_options:
            route_score = self.predict_route_efficiency({
                'distance': option['total_distance'],
                'cargo_weight': cargo_details['weight'],
                'temperature_requirements': cargo_details.get('required_temperature', 0),
                'fuel_cost': 1200,
                'weather_condition': 1,
                'port_congestion': 0.6,
                'season': 1,
                'cargo_type': 1
            })
            if route_score > best_transshipment_score:
                best_transshipment_score = route_score
                best_transshipment = option

        land_route = None
        if direct_score < 0 and best_transshipment_score < 0:
            land_route = ["Mumbai", "Delhi", "Rotterdam"]

        if direct_score >= best_transshipment_score and direct_score > 0:
            return {
                'type': 'direct',
                'route': direct_route,
                'score': direct_score,
                'distance': self.haversine_distance(origin, destination),
                'fuel_consumption': self.calculate_fuel_consumption(self.haversine_distance(origin, destination), cargo_details['weight'], ship_specs)
            }
        elif best_transshipment_score > 0:
            return {
                'type': 'transshipment',
                'route': best_transshipment,
                'score': best_transshipment_score,
                'hub': best_transshipment['hub'],
                'main_route': [origin, best_transshipment['hub']],
                'feeder_route': [best_transshipment['hub'], destination]
            }
        else:
            return {'type': 'land_route', 'route': land_route, 'reason': 'No viable maritime route found'}

class RoutePlanner:
    def __init__(self):
        self.optimizer = MaritimeRouteOptimizer()
        self.port_data = self.load_port_data()

    def load_port_data(self) -> Dict:
        return {
            "Jawaharlal Nehru Port Trust (JNPT) – Nhava Sheva": {"coordinates": (18.9519, 72.9324)},
            "Mundra Port": {"coordinates": (22.7680, 69.6729)},
            "Pipavav Port (APM Terminals Pipavav)": {"coordinates": (20.9137, 71.4533)},
            "Kandla Port (Deendayal Port)": {"coordinates": (23.0300, 70.2200)},
            "Cochin Port": {"coordinates": (9.9312, 76.2673)},
            "Mumbai Port": {"coordinates": (19.0760, 72.8777)},
            "Chennai Port": {"coordinates": (13.0827, 80.2707)},
            "Kolkata Port (including Haldia)": {"coordinates": (22.5726, 88.3639)},
            "Visakhapatnam Port": {"coordinates": (17.6868, 83.2185)},
            "Krishnapatnam Port": {"coordinates": (14.2543, 80.1144)},
            "Paradip Port": {"coordinates": (20.3160, 86.6100)},
            "Tuticorin Port (V.O. Chidambaranar Port)": {"coordinates": (8.7642, 78.1348)},
            "Goa Port (Mormugao Port)": {"coordinates": (15.4025, 73.7973)},
            "New Mangalore Port": {"coordinates": (12.9141, 74.8160)},
            "Ennore Port (Kamarajar Port)": {"coordinates": (13.2300, 80.3300)},
            "Port Blair": {"coordinates": (11.6234, 92.7265)},
            "APM Terminals Rotterdam": {"coordinates": (51.9225, 4.4792)}
        }

    def check_cargo_compatibility(self, cargo_details: Dict, ship_specs: Dict) -> bool:
        return cargo_details['weight'] <= ship_specs['max_capacity']

    def calculate_eta(self, route: Dict, weather_forecast: Dict, traffic_data: Dict) -> float:
        distance = route['distance']
        speed = 20
        return distance / (speed * 1.852)

    def calculate_fuel_efficiency(self, route: Dict) -> float:
        return 0.8

    def check_temperature_maintenance(self, route: Dict, required_temp: float) -> bool:
        return True

    def estimate_transshipment_time(self, hub: str, cargo_details: Dict) -> float:
        return 24

    def suggest_feeder_vessels(self, hub: str, destination: str, cargo_details: Dict) -> List[Dict]:
        return [{'name': 'Feeder Vessel 1', 'capacity': 1000}, {'name': 'Feeder Vessel 2', 'capacity': 1500}]

    def plan_route(self, origin: str, destination: str, cargo_details: Dict, ship_specs: Dict) -> Dict:
        origin_coords = self.port_data[origin]['coordinates']
        dest_coords = self.port_data[destination]['coordinates']

        if not self.check_cargo_compatibility(cargo_details, ship_specs):
            return {'error': 'Cargo incompatible with vessel specifications'}

        weather_forecast = {'condition': 'good'}
        traffic_data = {'density': 'low'}

        optimal_route = self.optimizer.find_optimal_route(origin_coords, dest_coords, cargo_details, ship_specs)

        if optimal_route['type'] != 'land_route':
            optimal_route.update({
                'weather_forecast': weather_forecast,
                'traffic_conditions': traffic_data,
                'estimated_time': self.calculate_eta(optimal_route, weather_forecast, traffic_data),
                'fuel_efficiency_score': self.calculate_fuel_efficiency(optimal_route),
                'temperature_maintenance': self.check_temperature_maintenance(optimal_route, cargo_details.get('required_temperature'))
            })

            if optimal_route['type'] == 'transshipment':
                optimal_route.update({
                    'transshipment_time': self.estimate_transshipment_time(optimal_route['hub'], cargo_details),
                    'feeder_vessel_suggestions': self.suggest_feeder_vessels(optimal_route['hub'], destination, cargo_details)
                })

        return optimal_route

if __name__ == "__main__":
    planner = RoutePlanner()

    cargo_details = {
        'weight': 5000,
        'type': 'container',
        'required_temperature': -18.0,
        'container_count': 250,
        'hazardous': False,
        'priority': 'normal'
    }

    ship_specs = {
        'max_capacity': 10000,
        'base_fuel_rate': 30,
        'reefer_capacity': 300,
        'vessel_type': 'container',
        'speed': 20
    }

    route = planner.plan_route(
        origin="Jawaharlal Nehru Port Trust (JNPT) – Nhava Sheva",
        destination="APM Terminals Rotterdam",
        cargo_details=cargo_details,
        ship_specs=ship_specs
    )

    print("\nOptimal Route Details:")
    print(f"Route Type: {route['type']}")
    if route['type'] != 'land_route':
        print(f"Total Distance: {route['distance']} km")
        print(f"Estimated Fuel Consumption: {route['fuel_consumption']} tons")
        print(f"Estimated Time: {route['estimated_time']} days")

        if route['type'] == 'transshipment':
            print(f"\nTransshipment Hub: {route['hub']}")
            print(f"Transshipment Time: {route['transshipment_time']} hours")
            print("\nFeeder Vessel Options:")
            for vessel in route['feeder_vessel_suggestions']:
                print(f"- {vessel['name']} ({vessel['capacity']} TEU)")
    else:
        print(f"Suggested Land Route: {route['route']}")
        print(f"Reason: {route['reason']}")