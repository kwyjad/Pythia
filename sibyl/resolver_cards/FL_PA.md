How this question resolves (checked against pythia/tools/compute_resolutions.py):
- Source: people-affected figures in the Resolver's facts tables. IFRC GO (field reports, DREF and emergency appeals) ranks above IDMC; a direct "affected" count ranks above a displacement figure. A legacy EM-DAT table is a last fallback, but no connector fills it today.
- A month with NO such record stays UNRESOLVED. It is never scored as zero. So p_zero for this question means "zero or no record".
- A record exists when a National Society reports or asks for funds, or IDMC logs displacement. Large floods handled with national means, or reported only in the press, may leave no record at all.
- Satellite exposure estimates (GDACS "population exposed") are a different quantity and never resolve this question.
