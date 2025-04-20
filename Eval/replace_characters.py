def replace_json_escape_characters(input_string):
    if input_string is None:
        return None  # Return None if the input is None
    
    # Replacement dictionary for JSON escape characters
    replacements = {
        r'\\': '\\',  # Backslash
        r'\"': '"',   # Double quote
        r'\b': '\b',  # Backspace
        r'\f': '\f',  # Form feed
        r'\n': '\n',  # New line
        r'\r': '\r',  # Carriage return
        r'\t': '\t'   # Horizontal tab
    }
    
    # Replace each escape character using the replacements dictionary
    for old, new in replacements.items():
        input_string = input_string.replace(old, new)
    
    return input_string

# Example usage:
json_string = 'Trajectories of CD4<sup>+</sup>/CD8<sup>+</sup> T-Cells Ratio 96 Weeks after Switching to Dolutegravir-Based Two-Drug Regimens: Results from a Multicenter Prospective Cohort Study. The aim of the present study was to evaluate CD4/CD8 dynamics in patients on dolutegravir (DTG)-based two-drug regimens (2DRs) and compare them with DTG-containing triple-drug regimens (3DRs). A prospective observational study was performed in the context of the SCOLTA cohort. Experienced PWH with HIV-RNA < 50 copies/mL were included if they were on the DTG-2DR, the DTG + tenofovir/emtricitabine (TDF/FTC) regimen, the DTG + tenofovir alafenamide (TAF)/FTC regimen, or the DTG + abacavir/lamivudine (ABC/3TC) regimen; they were followed-up for at least one year. A total of 533 PWH were enrolled, 120 in the DTG + 3TC group, 38 in the DTG + protease inhibitors (PI) group, 67 in the DTG + rilpivirine (RPV) group, 49 in the DTG + TDF/FTC group, 27 in the DTG + TAF/FTC group, and 232 in the DTG + ABC/3TC group. After one year, the CD4/CD8 ratio significantly increased in the PWH treated with DTG + 3TC (+0.08 ± 0.26), DTG + TDF/FTC (+0.1 ± 0.19), and DTG + ABC/3TC (+0.08 ± 0.25). At two years, the CD4/CD8 increase was confirmed for PWH on DTG + TDF/FTC (+0.16 ± 0.28) and DTG + ABC/3TC (+0.1 ± 0.3). In the SCOLTA cohort, PWH on 2DRs experienced a CD4/CD8 increase only in the DTG + 3TC group. Controlled studies with longer follow-up will clarify the long-term immunological and clinical impacts of DTG-2DR.'
result = replace_json_escape_characters(json_string)
print(result)
