# EU_Comparator
App that compares votes of different parliament members in European Parliament.

Data Source: https://github.com/HowTheyVote/data/releases

## Running the application

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Use the sidebar to select up to twenty entities (individual Members, political
   groups, or national delegations). You can narrow the scope by main votes,
   OEIL subjects, geographic tags, or a custom date range. The main panel
   summarises agreement counts and percentages, highlights the most aligned and
   divergent pair, lets you adjust the agreement trend bucket (monthly,
   quarterly, or weekly), and lists every shared vote with an enhanced search
   that spans titles, procedures, subjects, and geography labels.
