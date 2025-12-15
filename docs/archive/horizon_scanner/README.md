QHS AI-Powered Country Risk Analysis Bot
1.	Overview
This is an automated bot designed to generate comprehensive country risk analysis reports. It uses Google's Gemini 2.5 Pro AI model to perform detailed, real-time research on a list of specified countries.
The bot is built with Python and runs entirely within GitHub using GitHub Actions, requiring no local setup to operate once configured. The final output is a single Markdown (.md) file containing a summary table of all scenarios followed by the detailed, narrative reports for each country.
2.	How It Works
The process is managed by a single Python script (horizon_scanner.py) and is triggered by a GitHub Actions workflow.
Read Country List: The script begins by reading a simple list of country names from the hs_country_list.txt file.
Generate Individual Reports: For each country in the list, the script sends a detailed prompt (from hs_prompt.py) to the Gemini 2.5 Pro AI model. The AI conducts online research and generates a detailed narrative risk analysis. Crucially, it also embeds a hidden, machine-readable JSON data block within the report that summarizes the key scenario data.
Parse and Aggregate: After all individual reports are generated, the script uses a reliable Python-based parser (not another AI call) to find and extract the data from every hidden JSON block.
Build Final Report: The script then uses this extracted data to build a 100% accurate summary table. It combines this table with the individual narrative reports into a single, comprehensive Risk_Report.md file.
Upload Artifact: The GitHub Actions workflow saves this final Risk_Report.md file as an "artifact," making it available for download directly from the workflow's run page.
3.	Setup
To use this repository, you only need to perform one setup step:
Add the API Key Secret:
In the GitHub repository, go to Settings > Secrets and variables > Actions.
Click the "New repository secret" button.
Create a secret with the following name and value:
Name: GEMINI_API_KEY
Value: Paste in your Google AI API key.
4.	Usage
To generate a new report, follow these steps:
Update Country List:
Open the hs_country_list.txt file.
Add or remove countries as needed. The bot expects one country name per line. Lines starting with # will be ignored.
Run the GitHub Action:
Go to the "Actions" tab of the repository.
In the left sidebar, click on the "Generate Risk Report" workflow.
Click the "Run workflow" dropdown button on the right side of the screen, and then click the green "Run workflow" button that appears.
Monitor and Download:
The bot will now run. The process can take a significant amount of time depending on the number of countries (approximately 1.5 - 2 minutes per country).
You can click on the in-progress run to view the live logs and see which country is currently being processed.
Once the run is complete, the status will show a green checkmark. On the summary page for that run, an "Artifacts" section will appear at the bottom.
Click on "Risk-Analysis-Report" to download the final .md file.