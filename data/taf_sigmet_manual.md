# TAF and SIGMET Operational Manual

This handbook outlines the standard aviation weather decoding procedures for TAF (Terminal Aerodrome Forecasts) and SIGMET (Significant Meteorological Information) advisories.

## 1. Terminal Aerodrome Forecasts (TAF)

A TAF is a concise statement of the expected meteorological conditions at an airport during a specified period (usually 24 or 30 hours).

### Key Indicators:
- **TAF**: Identifies the message as a Terminal Aerodrome Forecast.
- **ICAO Ident**: 4-letter station identifier (e.g. `KDEN`, `KJFK`, `EGLL`).
- **Date/Time**: Issuance time in UTC (e.g. `142340Z` representing the 14th day of the month at 23:40 UTC).
- **Valid Period**: Start and end date/hour in UTC (e.g. `1500/1606` representing from the 15th at 00:00 UTC to the 16th at 06:00 UTC).

### Change Groups:
- **FM (From)**: Used to indicate a significant and rapid change occurring at a specific time (Format: `FM150400` representing From the 15th at 04:00 UTC). All conditions following are expected to replace preceding conditions.
- **TEMPO (Temporary)**: Used to indicate temporary fluctuations in meteorological conditions lasting less than one hour in each instance, and covering less than half of the sub-period. (e.g. `TEMPO 1502/1506 3/4SM -SN` indicates temporary drops to 3/4 statute mile visibility with light snow between 02:00 UTC and 06:00 UTC).
- **BECMG (Becoming)**: Used to indicate a gradual change in weather conditions over a specified period (usually 1 to 2 hours).
- **PROB30 / PROB40 (Probability)**: Indicates a 30% or 40% probability of temporary occurrences of hazardous conditions (e.g. `PROB30 1508/1512 1/2SM TSRA` indicates a 30% chance of 1/2SM visibility and thunderstorms between 08:00 UTC and 12:00 UTC).

### Special Elements:
- **Wind Shear (WS)**: Forecast wind shear below 2,000 feet (Format: `WS020/36045KT` representing Wind Shear at 2,000 feet, winds from 360° at 45 knots). This is an immediate flight safety warning.

---

## 2. Significant Meteorological Information (SIGMET)

A SIGMET is an advisory issued by a Meteorological Watch Office (MWO) concerning the occurrence or expected occurrence of specified en-route weather phenomena which may affect the safety of all aircraft operations.

### Types of Hazards Reported:
1. **Severe Turbulence (SEV TURB)**: High risk of loss of aircraft altitude control, structural strain, or passenger injury. Immediate flight routing alteration required.
2. **Severe Icing (SEV ICE / FZRA)**: Immediate icing hazard due to freezing rain or supercooled water droplets freezing onto control surfaces.
3. **Severe Thunderstorms (TS / SQL TS / FRQ TS)**: Thunderstorm lines or areas accompanied by heavy precipitation, hail, extreme turbulence, and wind shear.
4. **Volcanic Ash (VA / ASH)**: Volcanic dust clouds. Jet engine ingestion can cause immediate compressor stalls and complete engine failure. Avoidance is mandatory.
5. **Dust Storms / Sandstorms (DS / SS)**: Severe reductions in visibility.

### Identifiers:
- **WS / WV / WC**: SIGMET header codes (WS: Weather SIGMET, WV: Volcanic Ash, WC: Tropical Cyclone).
- **SIGMET Sequence**: Named sequentially using alphabet names (e.g. `SIGMET PAPA 1` or `SIGMET NOVEMBER 3`).
