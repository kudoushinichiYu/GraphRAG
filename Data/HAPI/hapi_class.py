from . import helper
import pandas as pd
import pycountry


class HapiClass:
    '''
    This class will store all data from HAPI
    '''
    LIMIT = 1000
    APP_IDENTIFIER = "U2ltb24gV2FuZzpzaHVyZW4wNDE5QDE2My5jb20="

    def __init__(self, location, country_name=None):
        '''
        :param location: Should be a specified Country, a capital string of length 3
        '''
        self.LOCATION = location
        if country_name:
            self.country_name = country_name
        else:
            self.country_name = pycountry.countries.get(alpha_3=self.LOCATION).name

        # Store data sets
        self.humanitarian_data = None
        self.refugee_data = None
        self.conflict_event_data = None

        # Initialize and get Risk Parameters
        self.global_rank = None
        self.vulnerability_risk = None
        self.risk_class = None
        self.coping_capacity_risk = None
        self.overall_risk = None
        self.hazard_exposure_risk = None
        # self.get_national_risk_data()

        # Poverty Rate (Only 2010 and 2015, hard to use)
        self.poverty_rate_data = None

        # Population
        self.population_data = None

        # Funding
        self.funding_data = None

    def get_humanitarian_needs_data(self):
        '''
        Retrieve humanitarian need data from HAPI, and store it in self.humanitarian_data
        Also drop useless columns
        :return: None
        '''
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'affected-people/humanitarian-needs', self.LOCATION)
        results = helper.fetch_data(base_url, HapiClass.LIMIT)
        # # results = pd.read_csv('humanitarian_needs.csv')
        # results = results.drop(['location_ref',
        #                               'location_name',
        #                               'location_code',
        #                               'admin1_is_unspecified',
        #                               'admin2_is_unspecified',
        #                               'admin1_ref',
        #                               'admin2_ref',
        #                               'admin1_code',
        #                               'admin2_code',
        #                               'admin1_name',
        #                               'admin2_name',
        #                               'min_age',
        #                               'max_age',
        #                               'resource_hdx_id'], axis=1)

        # Note for 'population_status' column
        # The PIN should not be summed across sectors or population statuses, as the same people can be present across multiple groups
        # For the number of people affected across all sectors, please use the PIN value where sector=intersectoral.
        # An “all” value in the gender, age_range, disable_marker, and population_group columns indicates no disaggregation
        self.humanitarian_data = results

    def get_refugee_data(self):
        '''
        Retrieve humanitarian need data from HAPI, and store it in self.refugee_data
        :return: None
        '''
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'affected-people/refugees-persons-of-concern', self.LOCATION)
        results = helper.fetch_data(base_url, HapiClass.LIMIT)
        print("Retrieves Refugee Data")
        self.refugee_data = results

    def get_national_risk_data(self):
        '''
        Retrieve national risk data from HAPI (one line)
        :return: None
        '''
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'coordination-context/national-risk', self.LOCATION)
        results = helper.fetch_data(base_url, HapiClass.LIMIT)
        # Get risk_class
        self.risk_class = results['risk_class'][0]

        # Get global_rank
        self.global_rank = results['global_rank'][0]

        # Get overall_risk
        self.overall_risk = results['overall_risk'][0]

        # Get hazard_exposure_risk
        self.hazard_exposure_risk = results['hazard_exposure_risk'][0]

        # Get vulnerability_risk
        self.vulnerability_risk = results['vulnerability_risk'][0]

        # Get coping_capacity_risk
        self.coping_capacity_risk = results['coping_capacity_risk'][0]

    def get_conflict_event_data(self):
        '''
        Retrieve conflict event data from HAPI
        :return: None
        '''
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'coordination-context/conflict-events', self.LOCATION)
        results = helper.fetch_data(base_url, HapiClass.LIMIT)
        # results = results.drop(['location_ref',
        #                         'location_name',
        #                         'location_code',
        #                         'admin1_is_unspecified',
        #                         'admin2_is_unspecified',
        #                         'admin1_ref',
        #                         'admin2_ref',
        #                         'admin1_code',
        #                         'admin2_code',
        #                         'resource_hdx_id'], axis=1)

        self.conflict_event_data = results

    def get_poverty_rate_data(self):
        """
        Retrieve poverty rate data from HAPI
        :return: None
        """
        # only have data from 2010 and 2015 (hard to use)
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'population-social/poverty-rate', self.LOCATION)
        results = helper.fetch_data(base_url, HapiClass.LIMIT)
        self.poverty_rate_data = results

    def get_population_data(self):
        """
        Retrieve population data from HAPI
        :return: None
        """
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'population-social/population', self.LOCATION)
        results = helper.fetch_data(base_url, HapiClass.LIMIT)
        # results = results.drop(['location_ref',
        #                         'location_name',
        #                         'location_code',
        #                         'admin1_is_unspecified',
        #                         'admin2_is_unspecified',
        #                         'admin1_ref',
        #                         'admin2_ref',
        #                         'admin1_code',
        #                         'admin2_code',
        #                         'admin1_name',
        #                         'admin2_name',
        #                         'resource_hdx_id'], axis=1)
        self.population_data = results


    def get_funding_data(self):
        """
        Retrieve funding data from HAPI
        :return: None
        """
        base_url = helper.construct_url(HapiClass.APP_IDENTIFIER, 'coordination-context/funding', self.LOCATION)
        results = helper.fetch_data(base_url, self.LIMIT)
        # results.drop(['resource_hdx_id',
        #               'appeal_code',
        #               'location_ref',
        #               'location_code',
        #               'location_name'], axis=1)
        self.funding_data = results




