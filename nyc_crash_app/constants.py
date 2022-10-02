title = """
        <h1 style='text-align: center; margin-bottom: -20px;'>
        Hi PyConES 2022!
        </h1>
        """
subtitle_up = """
              <p style='text-align: center;'>
              See all this? It's all done in Python with <b>Streamlit</b>
              </p>
              """
subtitle_down = """
                <p style='text-align: center;'>
                During this short time, we will look together at what can be done to help us explore the data
                </p>
                """

base_query = f"SELECT crash_date, MONTHNAME(TO_CHAR(crash_date)) as MONTH_NAME, crash_time, borough, zip_code, latitude, longitude, \
              number_of_persons_injured, number_of_persons_killed,\
              number_of_pedestrians_injured, number_of_pedestrians_killed,\
              number_of_cyclist_injured, number_of_cyclist_killed,\
              number_of_motorist_injured, number_of_motorist_killed,\
              vehicle_type_code_1, vehicle_type_code_2,\
              contributing_factor_vehicle_1\
              from motor_vehicle_collisions"
distinct_contributing_factors_query = """
        SELECT DISTINCT CONTRIBUTING_FACTOR_VEHICLE_1 
        FROM motor_vehicle_collisions 
        ORDER BY CONTRIBUTING_FACTOR_VEHICLE_1 ASC
        """

distinct_borough_query = """
        SELECT DISTINCT BOROUGH 
        FROM motor_vehicle_collisions 
        ORDER BY borough ASC
        """

distinct_vehicle_type_query = """
        SELECT DISTINCT VEHICLE_TYPE_CODE_1 
        FROM motor_vehicle_collisions 
        ORDER BY VEHICLE_TYPE_CODE_1 ASC
        """

days_of_week_ordered = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

