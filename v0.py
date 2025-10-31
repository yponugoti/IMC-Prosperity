from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Observation, Trade, Symbol, Listing, ProsperityEncoder
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANT = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"

PARAMS = {
    Product.SPREAD1: {
        "default_spread_mean": 59051.43667,
        "default_spread_std": 138.2209241,
        "spread_sma_window": 1500,
        "spread_std_window": 45,
        "zscore_threshold": 9,
        "target_position": 58,
    },
    Product.SPREAD2: {
        "default_spread_mean": 30408.50667,
        "default_spread_std": 80.62450703,
        "spread_sma_window": 1500,
        "spread_std_window": 45,
        "zscore_threshold": 9,
        "target_position": 96,
    }
}

BASKET1_WEIGHTS = {
    Product.CROISSANT: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1
}

BASKET2_WEIGHTS = {
    Product.CROISSANT: 4,
    Product.JAMS: 2
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANT: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        
        position_limit = self.LIMIT[product]

        # buy orders
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # sell orders
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume
        
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order

        return buy_order_volume, sell_order_volume
    
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def get_synthetic_basket1_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        #Constants
        CROISSANTS_PER_BASKET = BASKET1_WEIGHTS[Product.CROISSANT]
        JAMS_PER_BASKET = BASKET1_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET1_WEIGHTS[Product.DJEMBES]

        synthetic_order_price = OrderDepth()
        
        # Calculate the best bid and ask for each component
        croissants_best_bid = max(order_depths[Product.CROISSANT].buy_orders.keys()) if order_depths[Product.CROISSANT].buy_orders else 0
        croissants_best_ask = min(order_depths[Product.CROISSANT].sell_orders.keys()) if order_depths[Product.CROISSANT].sell_orders else float('inf')
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float('inf')
        djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
        djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float('inf')

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = croissants_best_bid * CROISSANTS_PER_BASKET + jams_best_bid * JAMS_PER_BASKET + djembes_best_bid * DJEMBES_PER_BASKET
        implied_ask = croissants_best_ask * CROISSANTS_PER_BASKET + jams_best_ask * JAMS_PER_BASKET + djembes_best_ask * DJEMBES_PER_BASKET

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = order_depths[Product.CROISSANT].buy_orders[croissants_best_bid] // CROISSANTS_PER_BASKET
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_bid] // DJEMBES_PER_BASKET
            implied_bid_volume = min(croissants_bid_volume, jams_bid_volume, djembes_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float('inf'):
            croissants_ask_volume = -order_depths[Product.CROISSANT].sell_orders[croissants_best_ask] // CROISSANTS_PER_BASKET
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            djembes_ask_volume = -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask] // DJEMBES_PER_BASKET
            implied_ask_volume = min(croissants_ask_volume, jams_ask_volume, djembes_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        
        return synthetic_order_price
    
    def get_synthetic_basket2_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        #Constants
        CROISSANTS_PER_BASKET = BASKET2_WEIGHTS[Product.CROISSANT]
        JAMS_PER_BASKET = BASKET2_WEIGHTS[Product.JAMS]

        synthetic_order_price = OrderDepth()
        
        # Calculate the best bid and ask for each component
        croissants_best_bid = max(order_depths[Product.CROISSANT].buy_orders.keys()) if order_depths[Product.CROISSANT].buy_orders else 0
        croissants_best_ask = min(order_depths[Product.CROISSANT].sell_orders.keys()) if order_depths[Product.CROISSANT].sell_orders else float('inf')
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float('inf')

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = croissants_best_bid * CROISSANTS_PER_BASKET + jams_best_bid * JAMS_PER_BASKET
        implied_ask = croissants_best_ask * CROISSANTS_PER_BASKET + jams_best_ask * JAMS_PER_BASKET

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = order_depths[Product.CROISSANT].buy_orders[croissants_best_bid] // CROISSANTS_PER_BASKET
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            implied_bid_volume = min(croissants_bid_volume, jams_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float('inf'):
            croissants_ask_volume = -order_depths[Product.CROISSANT].sell_orders[croissants_best_ask] // CROISSANTS_PER_BASKET
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            implied_ask_volume = min(croissants_ask_volume, jams_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        
        return synthetic_order_price
    
    def convert_synthetic_basket1_orders(self, 
        synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANT: [],
            Product.JAMS: [],
            Product.DJEMBES: []
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket1_order_depth = self.get_synthetic_basket1_order_depth(order_depths)
        best_bid = (
            max(synthetic_basket1_order_depth.buy_orders.keys())
            if synthetic_basket1_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket1_order_depth.sell_orders.keys())
            if synthetic_basket1_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(order_depths[Product.CROISSANT].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANT].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANT,
                croissants_price,
                quantity * BASKET1_WEIGHTS[Product.CROISSANT]
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET1_WEIGHTS[Product.JAMS]
            )
            djembes_order = Order(
                Product.DJEMBES,
                djembes_price,
                quantity * BASKET1_WEIGHTS[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANT].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)
        
        return component_orders
    
    def convert_synthetic_basket2_orders(self, 
        synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANT: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket2_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        best_bid = (
            max(synthetic_basket2_order_depth.buy_orders.keys())
            if synthetic_basket2_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket2_order_depth.sell_orders.keys())
            if synthetic_basket2_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(order_depths[Product.CROISSANT].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANT].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANT,
                croissants_price,
                quantity * BASKET2_WEIGHTS[Product.CROISSANT]
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET2_WEIGHTS[Product.JAMS]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANT].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)

        return component_orders

    def execute_spread1_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC1, synthetic_bid_price, -execute_volume)]

            aggregate_orders = self.convert_synthetic_basket1_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders

            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC1, synthetic_ask_price, execute_volume)]

            aggregate_orders = self.convert_synthetic_basket1_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders

            return aggregate_orders
        
    def execute_spread2_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)]

            aggregate_orders = self.convert_synthetic_basket2_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders

            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)]

            aggregate_orders = self.convert_synthetic_basket2_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders

            return aggregate_orders
    
    def spread1_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any]):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None
        
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD1]["spread_std_window"]:
            return None
        else:
            spread_std = np.std(spread_data["spread_history"][-self.params[Product.SPREAD1]["spread_std_window"]:])

        if len(spread_data['spread_history']) == self.params[Product.SPREAD1]["spread_sma_window"]:
            spread_mean = np.mean(spread_data['spread_history'])
            spread_data['curr_mean'] = spread_mean
        elif len(spread_data['spread_history']) > self.params[Product.SPREAD1]["spread_sma_window"]:
            spread_mean = spread_data['curr_mean'] + ((spread - spread_data['spread_history'][0]) / self.params[Product.SPREAD1]["spread_sma_window"])
            spread_data["spread_history"].pop(0)
        else:
            spread_mean = self.params[Product.SPREAD1]["default_spread_mean"]

        zscore = (spread - spread_mean) / spread_std
    
        
        if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(-self.params[Product.SPREAD1]["target_position"], basket_position, order_depths)

        if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(self.params[Product.SPREAD1]["target_position"], basket_position, order_depths)

        
            # if (zscore < 0 and spread_data["prev_zscore"] > 0) or (zscore > 0 and spread_data["prev_zscore"] < 0) or spread_data["clear_flag"]:
            #     if basket_position == 0:
            #         spread_data["clear_flag"] = False
            #     else:
            #         spread_data["clear_flag"] = True
            #         return self.execute_spread_orders(0, basket_position, order_depths)

    
        spread_data["prev_zscore"] = zscore
        return None
    
    def spread2_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any]):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None
        
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD2]["spread_std_window"]:
            return None
        else:
            spread_std = np.std(spread_data["spread_history"][-self.params[Product.SPREAD2]["spread_std_window"]:])

        if len(spread_data['spread_history']) == self.params[Product.SPREAD2]["spread_sma_window"]:
            spread_mean = np.mean(spread_data['spread_history'])
            spread_data['curr_mean'] = spread_mean
        elif len(spread_data['spread_history']) > self.params[Product.SPREAD2]["spread_sma_window"]:
            spread_mean = spread_data['curr_mean'] + ((spread - spread_data['spread_history'][0]) / self.params[Product.SPREAD2]["spread_sma_window"])
            spread_data["spread_history"].pop(0)
        else:
            spread_mean = self.params[Product.SPREAD2]["default_spread_mean"]

        zscore = (spread - spread_mean) / spread_std
    
        
        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(-self.params[Product.SPREAD2]["target_position"], basket_position, order_depths)

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(self.params[Product.SPREAD2]["target_position"], basket_position, order_depths)

        
            # if (zscore < 0 and spread_data["prev_zscore"] > 0) or (zscore > 0 and spread_data["prev_zscore"] < 0) or spread_data["clear_flag"]:
            #     if basket_position == 0:
            #         spread_data["clear_flag"] = False
            #     else:
            #         spread_data["clear_flag"] = True
            #         return self.execute_spread_orders(0, basket_position, order_depths)

    
        spread_data["prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
            traderObject = {}
            if state.traderData != None and state.traderData != "":
                traderObject = jsonpickle.decode(state.traderData)

            result = {}
            conversions = 0

            if Product.SPREAD1 not in traderObject:
                traderObject[Product.SPREAD1] = {
                    "spread_history": [],
                    "prev_zscore": 0,
                    "clear_flag": False,
                    "curr_avg": 0,
                }

            if Product.SPREAD2 not in traderObject:
                traderObject[Product.SPREAD2] = {
                    "spread_history": [],
                    "prev_zscore": 0,
                    "clear_flag": False,
                    "curr_avg": 0,
                }

            basket1_position = state.position[Product.PICNIC_BASKET1] if Product.PICNIC_BASKET1 in state.position else 0
            basket2_position = state.position[Product.PICNIC_BASKET2] if Product.PICNIC_BASKET2 in state.position else 0

            spread1_orders = self.spread1_orders(state.order_depths, Product.PICNIC_BASKET1, basket1_position, traderObject[Product.SPREAD1])
            spread2_orders = self.spread2_orders(state.order_depths, Product.PICNIC_BASKET2, basket2_position, traderObject[Product.SPREAD2])

            if spread1_orders != None:
                result[Product.DJEMBES] = spread1_orders[Product.DJEMBES]
                result[Product.PICNIC_BASKET1] = spread1_orders[Product.PICNIC_BASKET1]

            if spread2_orders != None:
                result[Product.CROISSANT] = spread2_orders[Product.CROISSANT]
                result[Product.JAMS] = spread2_orders[Product.JAMS]
                result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]

            traderData = jsonpickle.encode(traderObject)

            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData
