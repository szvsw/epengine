"""A module to inject DDY files into IDF files."""

from enum import Enum
from typing import Literal

from archetypal.idfclass import IDF
from pydantic import BaseModel, Field


class DDYField(Enum):
    """An enumeration of the fields in a DDY file that can be injected into an IDF file."""

    site_location = "SITE:LOCATION"
    design_day = "SIZINGPERIOD:DESIGNDAY"
    weather_file_condition_type = "SIZINGPERIOD:WEATHERFILECONDITIONTYPE"
    run_period_dst = "RUNPERIODCONTROL:DAYLIGHTSAVINGTIME"
    site_precipitation = "SITE:PRECIPITATION"
    roof_irrigation = "ROOFIRRIGATION"
    schedule_file = "SCHEDULE:FILE"


DesignDayName = Literal[
    "Ann Htg 99.6% Condns DB",
    "Ann Htg 99% Condns DB",
    "Ann Hum_n 99.6% Condns DP=>MCDB",
    "Ann Hum_n 99% Condns DP=>MCDB",
    "Ann Htg Wind 99.6% Condns WS=>MCDB",
    "Ann Htg Wind 99% Condns WS=>MCDB",
    "Ann Clg .4% Condns DB=>MWB",
    "Ann Clg 1% Condns DB=>MWB",
    "Ann Clg 2% Condns DB=>MWB",
    "Ann Clg .4% Condns WB=>MDB",
    "Ann Clg 1% Condns WB=>MDB",
    "Ann Clg 2% Condns WB=>MDB",
    "Ann Clg .4% Condns DP=>MDB",
    "Ann Clg 1% Condns DP=>MDB",
    "Ann Clg 2% Condns DP=>MDB",
    "Ann Clg .4% Condns Enth=>MDB",
    "Ann Clg 1% Condns Enth=>MDB",
    "Ann Clg 2% Condns Enth=>MDB",
    "January .4% Condns DB=>MCWB",
    "February .4% Condns DB=>MCWB",
    "March .4% Condns DB=>MCWB",
    "April .4% Condns DB=>MCWB",
    "May .4% Condns DB=>MCWB",
    "June .4% Condns DB=>MCWB",
    "July .4% Condns DB=>MCWB",
    "August .4% Condns DB=>MCWB",
    "September .4% Condns DB=>MCWB",
    "October .4% Condns DB=>MCWB",
    "November .4% Condns DB=>MCWB",
    "December .4% Condns DB=>MCWB",
    "January 2% Condns DB=>MCWB",
    "February 2% Condns DB=>MCWB",
    "March 2% Condns DB=>MCWB",
    "April 2% Condns DB=>MCWB",
    "May 2% Condns DB=>MCWB",
    "June 2% Condns DB=>MCWB",
    "July 2% Condns DB=>MCWB",
    "August 2% Condns DB=>MCWB",
    "September 2% Condns DB=>MCWB",
    "October 2% Condns DB=>MCWB",
    "November 2% Condns DB=>MCWB",
    "December 2% Condns DB=>MCWB",
    "January 5% Condns DB=>MCWB",
    "February 5% Condns DB=>MCWB",
    "March 5% Condns DB=>MCWB",
    "April 5% Condns DB=>MCWB",
    "May 5% Condns DB=>MCWB",
    "June 5% Condns DB=>MCWB",
    "July 5% Condns DB=>MCWB",
    "August 5% Condns DB=>MCWB",
    "September 5% Condns DB=>MCWB",
    "October 5% Condns DB=>MCWB",
    "November 5% Condns DB=>MCWB",
    "December 5% Condns DB=>MCWB",
    "January 10% Condns DB=>MCWB",
    "February 10% Condns DB=>MCWB",
    "March 10% Condns DB=>MCWB",
    "April 10% Condns DB=>MCWB",
    "May 10% Condns DB=>MCWB",
    "June 10% Condns DB=>MCWB",
    "July 10% Condns DB=>MCWB",
    "August 10% Condns DB=>MCWB",
    "September 10% Condns DB=>MCWB",
    "October 10% Condns DB=>MCWB",
    "November 10% Condns DB=>MCWB",
    "December 10% Condns DB=>MCWB",
    "January .4% Condns WB=>MCDB",
    "February .4% Condns WB=>MCDB",
    "March .4% Condns WB=>MCDB",
    "April .4% Condns WB=>MCDB",
    "May .4% Condns WB=>MCDB",
    "June .4% Condns WB=>MCDB",
    "July .4% Condns WB=>MCDB",
    "August .4% Condns WB=>MCDB",
    "September .4% Condns WB=>MCDB",
    "October .4% Condns WB=>MCDB",
    "November .4% Condns WB=>MCDB",
    "December .4% Condns WB=>MCDB",
    "January 2% Condns WB=>MCDB",
    "February 2% Condns WB=>MCDB",
    "March 2% Condns WB=>MCDB",
    "April 2% Condns WB=>MCDB",
    "May 2% Condns WB=>MCDB",
    "June 2% Condns WB=>MCDB",
    "July 2% Condns WB=>MCDB",
    "August 2% Condns WB=>MCDB",
    "September 2% Condns WB=>MCDB",
    "October 2% Condns WB=>MCDB",
    "November 2% Condns WB=>MCDB",
    "December 2% Condns WB=>MCDB",
    "January 5% Condns WB=>MCDB",
    "February 5% Condns WB=>MCDB",
    "March 5% Condns WB=>MCDB",
    "April 5% Condns WB=>MCDB",
    "May 5% Condns WB=>MCDB",
    "June 5% Condns WB=>MCDB",
    "July 5% Condns WB=>MCDB",
    "August 5% Condns WB=>MCDB",
    "September 5% Condns WB=>MCDB",
    "October 5% Condns WB=>MCDB",
    "November 5% Condns WB=>MCDB",
    "December 5% Condns WB=>MCDB",
    "January 10% Condns WB=>MCDB",
    "February 10% Condns WB=>MCDB",
    "March 10% Condns WB=>MCDB",
    "April 10% Condns WB=>MCDB",
    "May 10% Condns WB=>MCDB",
    "June 10% Condns WB=>MCDB",
    "July 10% Condns WB=>MCDB",
    "August 10% Condns WB=>MCDB",
    "September 10% Condns WB=>MCDB",
    "October 10% Condns WB=>MCDB",
    "November 10% Condns WB=>MCDB",
    "December 10% Condns WB=>MCDB",
]

WeatherFileConditionType = Literal[
    "Summer Extreme",
    "Summer Typical",
    "Winter Extreme",
    "Winter Typical",
    "Autumn Typical",
    "Spring Typical",
]


class DDYSizingSpec(BaseModel):
    """A class to define how to inject a DDY file into an IDF file."""

    match: bool = Field(
        default=True,
        description="If True, will attempt to match design days adn weather condition types from the DDY to existing design days in the IDF file.",
    )
    design_days: list[DesignDayName] | Literal["All"] | None = Field(
        default=None,
        description="Additional design days to inject from the DDY into the IDF file.",
    )
    conditions_types: list[WeatherFileConditionType] | Literal["All"] | None = Field(
        default=None,
        description="Additional weather file condition types to inject from the DDY into the IDF file.",
    )
    raise_on_not_found: bool = Field(
        default=True,
        description="If True, will raise an error if a design day or weather file condition type is not found in the DDY file.",
    )

    def inject_ddy(self, idf: IDF, ddy: IDF):
        """Copies the DDY file into the IDF file according to the spec.

        Currently, only the following DDY fields are supported:
        - SITE:LOCATION
        - SIZINGPERIOD:DESIGNDAY
        - SIZINGPERIOD:WEATHERFILECONDITIONTYPE

        The following DDY fields are ignored as the just contain rain information or are not used:
        - RUNPERIODCONTROL:DAYLIGHTSAVINGTIME
        - SITE:PRECIPITATION
        - ROOFIRRIGATION
        - SCHEDULE:FILE


        Args:
            idf (IDF): The IDF file to inject the DDY file into.
            ddy (IDF): The DDY file to inject into the IDF file.
        """
        for key, sequence in ddy.idfobjects.items():
            if not sequence:
                continue

            # will raise an error if the key is not in the DDYKey enum to protect
            # against additional keys being added to ddys in the future
            field = DDYField(key)

            if field == DDYField.site_location:
                self.handle_site_location(idf, ddy)
            elif field == DDYField.design_day:
                self.handle_design_days(idf, ddy)
            elif field == DDYField.weather_file_condition_type:
                self.handle_weather_file_condition_types(idf, ddy)
            else:
                # skip dst, site_precipitation, roof_irrigation, and schedule_file for now
                pass

        del ddy

    def remove_and_replace(
        self, idf: IDF, ddy: IDF, field: DDYField, copy_names: set[str]
    ):
        """Removes all objects of the given field and replaces them with the new ones.

        Raises an error if the object is not found in the DDY file and `self.raise_on_not_found` is True.

        Args:
            idf (IDF): The IDF file to remove and replace objects from.
            ddy (IDF): The DDY file to copy objects from.
            field (DDYField): The field to remove and replace objects from.
            copy_names (set[WeatherFileConditionType] | set[DesignDayName]): The names of the objects to copy.
        """
        idf.removeallidfobjects(field.value)
        for obj_name in copy_names:
            obj = ddy.getobject(field.value, obj_name)
            if not obj and self.raise_on_not_found:
                raise DDYFieldNotFoundError(field=field, obj=obj_name)
            idf.addidfobject(obj)

    def handle_site_location(self, idf: IDF, ddy: IDF):
        """Handles the SITE:LOCATION field in the DDY file.

        Args:
            idf (IDF): The IDF file to inject the DDY file into.
            ddy (IDF): The DDY file to inject into the IDF file.
        """
        field = DDYField.site_location
        idf.removeallidfobjects(field.value)
        idf.addidfobjects(ddy.idfobjects[field.value])

    def handle_design_days(self, idf: IDF, ddy: IDF):
        """Handles the SIZINGPERIOD:DESIGNDAY field in the DDY file.

        Args:
            idf (IDF): The IDF file to inject the DDY file into.
            ddy (IDF): The DDY file to inject into the IDF file.
        """
        field = DDYField.design_day
        sequence = ddy.idfobjects[field.value]
        all_design_day_names = [d.Name for d in sequence]
        if self.design_days == "All":
            target_obj_names = all_design_day_names
        else:
            target_obj_names = (
                [
                    name
                    for name in all_design_day_names
                    if any(d in name for d in self.design_days)
                ]
                if self.design_days
                else []
            )
        if self.raise_on_not_found and not target_obj_names and self.design_days:
            raise DDYFieldNotFoundError(field=field, obj=", ".join(self.design_days))
        self.remove_and_replace(idf, ddy, field, set(target_obj_names))

    def handle_weather_file_condition_types(self, idf: IDF, ddy: IDF):
        """Handles the SIZINGPERIOD:WEATHERFILECONDITIONTYPE field in the DDY file.

        Args:
            idf (IDF): The IDF file to inject the DDY file into.
            ddy (IDF): The DDY file to inject into the IDF.
        """
        field = DDYField.weather_file_condition_type
        all_condition_types = [c.Name for c in ddy.idfobjects[field.value]]
        if self.conditions_types == "All":
            obj_names = all_condition_types
        elif isinstance(self.conditions_types, list):
            obj_names = self.conditions_types
        else:
            obj_names = []

        existing_sequence = idf.idfobjects[field.value]
        if self.match and existing_sequence:
            desired_objects = [obj.Name for obj in existing_sequence]
            obj_names.extend(desired_objects)
        self.remove_and_replace(idf, ddy, field, set(obj_names))


class DDYFieldNotFoundError(Exception):
    """Raised when a field is not found in a DDY file."""

    # excepts a field name and object name and raises an error

    def __init__(self, field: DDYField, obj: str):
        """Initialize the error.

        Args:
            field (DDYField): The field that was not found.
            obj (str): The object that was not found.
        """
        message = f"{field.value}::{obj} not found in ddy file."
        super().__init__(message)


if __name__ == "__main__":
    from pathlib import Path

    idf_path = Path("US+SF+CZ4A+oilfurnace+unheatedbsmt+IECC_2021.idf")
    epw_path = Path(
        "D:/onebuilding/WMO_Region_4_North_and_Central_America/USA_United_States_of_America/MA_Massachusetts/USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023/USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023.epw"
    )

    ddy_path = epw_path.with_suffix(".ddy")
    idf = IDF(
        idf_path.as_posix(),
        epw=epw_path.as_posix(),
    )  # pyright: ignore [reportArgumentType]
    ddy = IDF(
        ddy_path.as_posix(),
        as_version="9.2.0",
        file_version="9.2.0",
        prep_outputs=False,
    )

    ddyspec = DDYSizingSpec(
        design_days=["January 10% Condns WB=>MCDB"],
        conditions_types=["Summer Extreme"],
        match=True,
    )

    ddyspec.inject_ddy(idf, ddy)
