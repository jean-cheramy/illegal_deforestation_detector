from codecarbon import EmissionsTracker

# Initialize tracker
tracker = EmissionsTracker(allow_multiple_runs=True)

class EmissionsData:
    """Store energy and emissions data.

    Attributes:
        energy_consumed (float): Energy consumed in kWh.
        emissions (float): Emissions in kg CO2e.
    """
    def __init__(self, energy_consumed: float, emissions: float):
        self.energy_consumed = energy_consumed
        self.emissions = emissions


def clean_emissions_data(emissions_data: dict)-> dict:
    """Remove unwanted fields from emissions data"""
    data_dict = emissions_data.__dict__
    fields_to_remove = ['timestamp', 'project_name', 'experiment_id', 'latitude', 'longitude']
    return {k: v for k, v in data_dict.items() if k not in fields_to_remove}
