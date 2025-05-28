def format_flight_details(flight_data):
    """
    Converts flight data from a dictionary to a natural language string.
    """
    flight = flight_data['flights'][0]
    departure_airport = flight['departure_airport']
    arrival_airport = flight['arrival_airport']

    # Date and time formatting
    dep_datetime_str = departure_airport['time']
    arr_datetime_str = arrival_airport['time']

    # Assuming "YYYY-MM-DD HH:MM" format
    dep_date = dep_datetime_str.split(' ')[0]
    dep_time = dep_datetime_str.split(' ')[1]
    arr_date = arr_datetime_str.split(' ')[0] # Assuming same day arrival for this example
    arr_time = arr_datetime_str.split(' ')[1]


    description = f"This JSON describes a {flight_data['type'].lower()} flight itinerary.\\n\\n"
    description += (
        f"The flight departs from {departure_airport['name']} ({departure_airport['id']}) "
        f"on {dep_date} at {dep_time} and is scheduled to arrive at "
        f"{arrival_airport['name']} ({arrival_airport['id']}) "
        f"on {arr_date} at {arr_time}. " # Consider adding logic for arrival date if it can differ
        f"The flight duration is {flight['duration']} minutes.\\n\\n"
    )
    description += (
        f"The airline operating this flight is {flight['airline']}, "
        f"with flight number {flight['flight_number']}, and the aircraft is an {flight['airplane']}. "
        f"The travel class for this ticket is {flight['travel_class']}. "
    )
    if 'legroom' in flight and flight['legroom']:
        description += f"Passengers can expect legroom of {flight['legroom']}. "

    if 'ticket_also_sold_by' in flight and flight['ticket_also_sold_by']:
        sold_by_airlines = ", ".join(flight['ticket_also_sold_by'])
        description += f"Tickets for this flight may also be sold by {sold_by_airlines}. "

    if 'extensions' in flight and flight['extensions']:
        # Filter out carbon emissions related extensions
        other_extensions = [ext for ext in flight['extensions'] if "carbon emissions" not in ext.lower()]
        if other_extensions:
            extensions_str = '", "'.join(other_extensions)
            description += f"Additional features noted for this flight include \"{extensions_str}\".\\n\\n"
        else:
            description += "\\n\\n" # Add newline if no other extensions
    else:
        description += "\\n\\n"


    description += f"The total duration for this {flight_data['type'].lower()} trip is {flight_data['total_duration']} minutes. "
    description += f"The price of the ticket is {flight_data['price']} (the currency is not specified)."

    return description

if __name__ == '__main__':
    sample_data = {
      "flights": [
        {
          "departure_airport": {
            "name": "Hangzhou International Airport",
            "id": "HGH",
            "time": "2025-05-26 13:00"
          },
          "arrival_airport": {
            "name": "Beijing Capital International Airport",
            "id": "PEK",
            "time": "2025-05-26 15:15"
          },
          "duration": 135,
          "airplane": "Airbus A330",
          "airline": "Air China",
          "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/CA.png",
          "travel_class": "Economy",
          "flight_number": "CA 1713",
          "ticket_also_sold_by": [
            "Shandong"
          ],
          "legroom": "31 in",
          "extensions": [
            "Average legroom (31 in)",
            "In-seat power outlet",
            "Carbon emissions estimate: 100 kg"
          ]
        }
      ],
      "total_duration": 135,
      "carbon_emissions": {
        "this_flight": 101000,
        "typical_for_this_route": 121000,
        "difference_percent": -17
      },
      "price": 414,
      "type": "One way",
      "airline_logo": "https://www.gstatic.com/flights/airline_logos/70px/CA.png",
      "booking_token": "WyJDalJJZDB0d1ZFNVNjSE5pT1ZWQlFqRXpPRUZDUnkwdExTMHRMUzB0TFhaMGFtUXhORUZCUVVGQlIyZDJVVVl3U1cxeVMyZEJFZ1pEUVRFM01UTWFDd2lmd3dJUUFob0RWVk5FT0J4d244TUMiLFtbIkhHSCIsIjIwMjUtMDUtMjYiLCJQRUsiLG51bGwsIkNBIiwiMTcxMyJdXV0="
    }
    
    formatted_string = format_flight_details(sample_data)
    print(formatted_string) 