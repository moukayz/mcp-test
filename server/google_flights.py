import os
from typing import List, Optional, Self

from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP # Assuming this path is correct
from serpapi import GoogleSearch
import json
import logging
import dotenv 
dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class AirportInfo(BaseModel):
    name: str
    id: str
    time: str


class FlightModel(BaseModel):
    departure_airport: AirportInfo
    arrival_airport: AirportInfo
    duration: int
    airplane: str
    airline: str
    airline_logo: str
    travel_class: str
    flight_number: str
    extensions: List[str]
    ticket_also_sold_by: Optional[List[str]] = None
    legroom: Optional[str] = None
    overnight: Optional[bool] = None
    often_delayed_by_over_30_min: Optional[bool] = None
    plane_and_crew_by: Optional[str] = None


class LayoverModel(BaseModel):
    duration: int
    name: str
    id: str
    overnight: Optional[bool] = None


class FlightDetailsModel(BaseModel):
    flights: List[FlightModel]
    layovers: Optional[List[LayoverModel]] = None
    total_duration: int
    price: Optional[int] = None
    type: str
    airline_logo: Optional[str] = None # This can be present if multiple airlines
    extensions: Optional[List[str]] = None
    departure_token: Optional[str] = None
    booking_token: Optional[str] = None
    return_flights: Optional[List[Self]] = None


class PriceInsightsModel(BaseModel):
    lowest_price: Optional[int] = None
    price_level: Optional[str] = None
    typical_price_range: Optional[List[int]] = None
    price_history: Optional[List[List[int]]] = None


class FlightsResponseModel(BaseModel):
    best_flights: Optional[List[FlightDetailsModel]] = None
    other_flights: Optional[List[FlightDetailsModel]] = None
    price_insights: Optional[PriceInsightsModel] = None


class FlightSearchParams(BaseModel):
    departure_id: Optional[str] = None
    """Parameter defines the departure airport code or location kgmid.
    An airport code is an uppercase 3-letter code. You can search for it on Google Flights or IATA.
    For example, CDG is Paris Charles de Gaulle Airport and AUS is Austin-Bergstrom International Airport.
    A location kgmid is a string that starts with /m/. You can search for a location on Wikidata and use its "Freebase ID" as the location kgmid. For example, /m/0vzm is the location kgmid for Austin, TX.
    You can specify multiple departure airports by separating them with a comma. For example, CDG,ORY,/m/04jpl."""

    arrival_id: Optional[str] = None
    """Parameter defines the arrival airport code or location kgmid.
    An airport code is an uppercase 3-letter code. You can search for it on Google Flights or IATA.
    For example, CDG is Paris Charles de Gaulle Airport and AUS is Austin-Bergstrom International Airport.
    A location kgmid is a string that starts with /m/. You can search for a location on Wikidata and use its "Freebase ID" as the location kgmid. For example, /m/0vzm is the location kgmid for Austin, TX.
    You can specify multiple arrival airports by separating them with a comma. For example, CDG,ORY,/m/04jpl"""

    gl: Optional[str] = "us"
    """Parameter defines the country to use for the Google Flights search. It's a two-letter country code. (e.g., us for the United States, uk for United Kingdom, or fr for France) Head to the Google countries page for a full list of supported Google countries."""

    hl: Optional[str] = "en"
    """Parameter defines the language to use for the Google Flights search. It's a two-letter language code. (e.g., en for English, es for Spanish, or fr for French). Head to the Google languages page for a full list of supported Google languages."""

    currency: Optional[str] = "USD"
    """Parameter defines the currency of the returned prices. Default to USD. Head to the Google Travel Currencies page for a full list of supported currency codes."""

    type: Optional[int] = None
    """Parameter defines the type of the flights.
    Available options:
    1 - Round trip (default)
    2 - One way
    3 - Multi-city
    When this parameter is set to 3, use multi_city_json to set the flight information.
    To obtain the returning flight information for Round Trip (1), you need to make another request using a departure_token."""

    outbound_date: Optional[str] = None
    """Parameter defines the outbound date. The format is YYYY-MM-DD. e.g. 2025-05-23"""

    return_date: Optional[str] = None
    """Parameter defines the return date. The format is YYYY-MM-DD. e.g. 2025-05-29
    Parameter is required if type parameter is set to: 1 (Round trip)"""

    travel_class: Optional[int] = None
    """Parameter defines the travel class.
    Available options:
    1 - Economy (default)
    2 - Premium economy
    3 - Business
    4 - First"""

    adults: Optional[int] = None
    """Parameter defines the number of adults. Default to 1."""

    sort_by: Optional[int] = None
    """Parameter defines the sorting order of the results.
    Available options:
    1 - Top flights (default)
    2 - Price
    3 - Departure time
    4 - Arrival time
    5 - Duration
    6 - Emissions"""

    stops: Optional[int] = None
    """Parameter defines the number of stops during the flight.
    Available options:
    0 - Any number of stops (default)
    1 - Nonstop only
    2 - 1 stop or fewer
    3 - 2 stops or fewer"""

    exclude_airlines: Optional[str] = None
    """Parameter defines the airline codes to be excluded. Split multiple airlines with comma.
    It can't be used together with include_airlines.
    Each airline code should be a 2-character IATA code consisting of either two uppercase letters or one uppercase letter and one digit. You can search for airline codes on IATA.
    For example, UA is United Airlines.
    Additionally, alliances can be also included here:
    STAR_ALLIANCE - Star Alliance
    SKYTEAM - SkyTeam
    ONEWORLD - Oneworld
    exclude_airlines and include_airlines parameters can't be used together."""

    include_airlines: Optional[str] = None
    """Parameter defines the airline codes to be included. Split multiple airlines with comma.
    It can't be used together with exclude_airlines.
    Each airline code should be a 2-character IATA code consisting of either two uppercase letters or one uppercase letter and one digit. You can search for airline codes on IATA.
    For example, UA is United Airlines.
    Additionally, alliances can be also included here:
    STAR_ALLIANCE - Star Alliance
    SKYTEAM - SkyTeam
    ONEWORLD - Oneworld
    exclude_airlines and include_airlines parameters can't be used together."""

    bags: Optional[int] = None
    """Parameter defines the number of carry-on bags. Default to 0."""

    outbound_times: Optional[str] = None
    """Parameter defines the outbound times range. It's a string containing two (for departure only) or four (for departure and arrival) comma-separated numbers. Each number represents the beginning of an hour. For example:
    4,18: 4:00 AM - 7:00 PM departure
    0,18: 12:00 AM - 7:00 PM departure
    19,23: 7:00 PM - 12:00 AM departure
    4,18,3,19: 4:00 AM - 7:00 PM departure, 3:00 AM - 8:00 PM arrival
    0,23,3,19: unrestricted departure, 3:00 AM - 8:00 PM arrival"""

    return_times: Optional[str] = None
    """Parameter defines the return times range. It's a string containing two (for departure only) or four (for departure and arrival) comma-separated numbers. Each number represents the beginning of an hour. For example:
    4,18: 4:00 AM - 7:00 PM departure
    0,18: 12:00 AM - 7:00 PM departure
    19,23: 7:00 PM - 12:00 AM departure
    4,18,3,19: 4:00 AM - 7:00 PM departure, 3:00 AM - 8:00 PM arrival
    0,23,3,19: unrestricted departure, 3:00 AM - 8:00 PM arrival
    Parameter should only be used when type parameter is set to: 1 (Round trip)"""

    layover_duration: Optional[str] = None
    """Parameter defines the layover duration, in minutes. It's a string containing two comma-separated numbers. For example, specify 90,330 for 1 hr 30 min - 5 hr 30 min."""

    exclude_conns: Optional[str] = None
    """Parameter defines the connecting airport codes to be excluded.
    An airport ID is an uppercase 3-letter code. You can search for it on Google Flights or IATA.
    For example, CDG is Paris Charles de Gaulle Airport and AUS is Austin-Bergstrom International Airport.
    You can also combine multiple Airports by joining them with a comma (value + , + value; eg: CDG,AUS)."""

    max_duration: Optional[int] = None
    """Parameter defines the maximum flight duration, in minutes. For example, specify 1500 for 25 hours."""

    departure_token: Optional[str] = None

def create_search_params(params: FlightSearchParams) -> dict:
    res = params.model_dump(exclude_none=True)
    res["api_key"] = GOOGLE_FLIGHTS_API_KEY
    res["engine"] = "google_flights"
    logger.info(f"Flight search params: {res}")

    return res

def call_search_api(params: FlightSearchParams) -> FlightsResponseModel | str:
    try:
        search = GoogleSearch(create_search_params(params))
        results_dict = search.get_dict()
    except Exception as e:
        logger.error(f"Error calling Google Flights API: {e}")
        return f"Error calling Google Flights API: {e}"

    print_debug_info(results_dict)

    if "other_flights" not in results_dict:
        return f"API Error: result: {results_dict}"


    try:
        response_model = FlightsResponseModel.model_validate(results_dict)
    except Exception as e: # Handles PydanticValidationError, etc.
        logger.error(f"Error parsing outbound API response: {e}. Raw results snippet: {str(results_dict)[:500]}")
        return f"Error parsing API response: {e}. Raw results snippet: {str(results_dict)[:500]}"

    return response_model


# Initialize FastMCP server
mcp = FastMCP("google_flights")

GOOGLE_FLIGHTS_API_KEY = os.getenv("SERPAPI_API_KEY")

def print_debug_info(results_dict: dict):
    if "search_parameters" in results_dict:
        search_parameters = results_dict["search_parameters"]
        logger.info(f"returned flight search parameters: {search_parameters}")

    if "search_metadata" in results_dict:
        json_endpoint = results_dict["search_metadata"].get("json_endpoint", None)
        logger.info(f"Flight search JSON endpoint: {json_endpoint}")

def get_return_flights(flight_details: FlightDetailsModel, search_params: FlightSearchParams, max_results: int) -> List[FlightDetailsModel]:
    if not flight_details.departure_token:
        return []

    logger.info(f"Getting return flights for {flight_details.departure_token}")
    search_params.departure_token = flight_details.departure_token
    response = call_search_api(search_params)
    if isinstance(response, str):
        return []

    if response.best_flights:
        return response.best_flights[:max_results]
    elif response.other_flights:
        return response.other_flights[:max_results]
    else:
        return []

@mcp.tool()
async def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,  # Format: YYYY-MM-DD
    type: int,  # 1 for Round trip, 2 for One way
    return_date: Optional[str] = None,  # Format: YYYY-MM-DD, required if type is 1,
    max_results: Optional[int] = 10
) -> str:
    """
    Searches for flight information using the Google Flights API via SerpApi.

    This tool queries the Google Flights engine to find the best and other available
    flight options based on the provided departure and arrival locations, dates,
    and trip type. 

    Args:
        departure_id: The departure airport code (e.g., "SFO", "JFK") or a
                      Google KG Midtown ID (e.g., "/m/0vzm" for Austin, TX).
                      This specifies the origin of the flight.
        arrival_id: The arrival airport code (e.g., "LAX", "CDG") or a
                    Google KG Midtown ID (e.g., "/m/04jpl" for London).
                    This specifies the destination of the flight.
        outbound_date: The date of the outbound flight, formatted as YYYY-MM-DD.
                       (e.g., "2024-12-25").
        type: An integer indicating the type of trip.
              - 1: Round trip. If selected, `return_date` is also required.
                   The function will attempt to fetch return flights.
              - 2: One-way trip.
        return_date: The date of the return flight for round trips, formatted as
                     YYYY-MM-DD (e.g., "2025-01-05"). This parameter is
                     required if `type` is 1, and ignored otherwise.
        max_results: The maximum number of search flight results to return for each
                     category (e.g., best outbound, other outbound, best return, etc.). Default is 10.

    Returns:
        A list of flight objects. Each flight object is the departure flight, 
        and if the trip is a round trip, its return_flights field will also include the return flights of the departure flight.
        note the return size is controlled by the max_results parameter.
        if the type is round trip, and if max_results is 2, in the returned results there are 2 departure flights, and each of them has 2 return flights,
        So there are totally 4 round trip flights.
        if the type is one way, the returned results will only include the departure flights.
    """
    if type == 1 and not return_date:
        return "Error: Return date is required for round trip flights (type=1)."

    params_outbound = FlightSearchParams(
        departure_id=departure_id,
        arrival_id=arrival_id,
        outbound_date=outbound_date,
        type=type,
    )

    if type == 1 and return_date:
        params_outbound.return_date = return_date

    response_model_outbound = call_search_api(params_outbound)
    if isinstance(response_model_outbound, str):
        return response_model_outbound

    total_flights = (response_model_outbound.best_flights or []) + (response_model_outbound.other_flights or [])
    departure_flights: List[FlightDetailsModel] = total_flights[:max_results]

    if not departure_flights:
        return "No flight data (best_flights or other_flights) found in the API response for the outbound journey."

    if type == 1:
        for flight_detail in departure_flights:
            return_flights = get_return_flights(flight_detail, params_outbound, max_results)
            flight_detail.return_flights = return_flights

    output_data = [flight.model_dump(exclude_none=True) for flight in departure_flights]

    try:
        json_output = json.dumps(output_data)
        return json_output
    except Exception as e:
        logger.error(f"Error formatting output data to JSON: {e}")
        return f"Error formatting output data to JSON: {e}"

if __name__ == "__main__":
    # This allows running the MCP server directly for this tool
    # Ensure that mcp.server.fastmcp is accessible in your PYTHONPATH
    # Example command to run: python server/google_flights.py
    mcp.run(transport='stdio')
