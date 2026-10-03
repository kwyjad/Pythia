How this question resolves (checked against pythia/tools/compute_resolutions.py):
- Source: ACLED's monthly count of reported fatalities for the country, summed over ALL event types (battles, violence against civilians, explosions/remote violence, riots, protests). Not battle deaths alone.
- Only complete months count: a month's figure is read once the month has ended and the row has been refreshed after it.
- A month ACLED covered in which it recorded no fatal event for the country resolves to ZERO.
- The first resolution is read about four weeks after the month ends. ACLED keeps adding events to a month for weeks after that, so early totals run a little low; later revisions are kept but do not change the first score.
