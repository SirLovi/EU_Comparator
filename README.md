# EU_Comparator
Script that compares votes of different parliament members in European Parliament.

## Running the application

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Use the sidebar to select up to six Members of the European Parliament. The main
   panel summarises how often they voted the same way and lists each shared vote
   alongside the individual positions.

Tables
members.csv
Each row represents a Member of the European Parliament (MEP).

Column	Type	Description
id	integer	Member ID as used by the MEP Directory.
first_name	string	First name
last_name	string	Last name
country_code	string	3-letter ISO-3166-1 code
date_of_birth	date (optional)	Date of birth
email	string (optional)	Email address
facebook	string (optional)	Facebook profile URL
twitter	string (optional)	Twitter account URL
countries.csv
Each row represents an EU member state.

Column	Type	Description
code	string	3-letter ISO-3166-1 code
iso_alpha_2	string	2-letter ISO-3166-1 code
label	string	Label as published by the Publications Office of the European Union
groups.csv
Each row represents a political group in the European Parliament.

Column	Type	Description
code	string	Unique identifier for the political group
official_label	string	Official label as published by the Publications Office of the European Union
label	string	Label based on the official label. Prefixes and suffixes such as "Group" are removed for clarity.
short_label	string	Short label or abbreviation
group_memberships.csv
Each row represents a membership of an MEP in a political group.

MEPs can change their political group during the term, i.e., each MEP is part of one or more political groups over the course of a term. Non-attached MEPs are a member of the NI group.

Column	Type	Description
member_id	integer	Member ID
group_code	string	Group code
term	integer	Parliamentary term
start_date	date	Start date
end_date	date (optional)	End date. If empty, the MEP the membership is still active.
votes.csv
Each row represents a roll-call vote in plenary.

Column	Type	Description
id	integer	Vote ID
timestamp	dateTime	Date and time of the vote
display_title	string (optional)	Title that can be used to refer to the vote. In most cases, this is the title published in the roll-call vote results. If the title in the roll-call vote results is empty, this falls back to the procedure title.
reference	string (optional)	Reference to a plenary document such as a report or a resolution
description	string (optional)	Description of the vote as published in the roll-call vote results
is_main	boolean	Whether this vote is a main vote. We classify certain votes as main votes based on the text description in the voting records published by Parliament. For example, if Parliament has voted on amendments, only the vote on the text as a whole is classified as a main vote. Certain votes such as votes on the agenda are not classified as main votes. This is not an official classification by the European Parliament and there may be false negatives.
procedure_reference	string (optional)	Procedure reference as listed in the Legislative Observatory
procedure_title	string (optional)	Title of the legislative procedure as listed in the Legislative Observatory
procedure_type	string (optional)	Procedure type as listed in the Legislative Observatory. This is a 3-letter code such as COD, RSP, or BUD.
procedure_stage	string (optional)	Stage of the procedure in which the vote took place. One of OLP_FIRST_READING, OLP_SECOND_READING, OLP_THIRD_READING.This field is only available for votes starting in 2024 and if the vote is part of an Ordinary Legislative Procedure.
count_for	integer	Number of MEPs who voted in favor
count_against	integer	Number of MEPs who voted against
count_abstention	integer	Number of MEPs who abstained
count_did_not_vote	integer	Number of MEPs who didn’t participate in the vote
result	string (optional)	Vote result. One of ADOPTED, REJECTED, LAPSED. This field is only available for votes starting in 2024.
member_votes.csv
Each row represents how an MEP voted in a roll-call vote.

Column	Type	Description
vote_id	integer	Vote ID
member_id	integer	Member ID
position	string	Vote position. One of FOR, AGAINST, ABSTENTION if the MEP participated in the vote or DID_NOT_VOTE if the MEP wasn’t present for the vote. We currently do not differentiate between MEPs who did not vote with or without an excuse.
country_code	string	Country code
group_code	string (optional)	Group code. This references the political group that the MEP was part of on the day of the vote. This is not necessarily the MEP’s current political group.
eurovoc_concepts.csv
Each row represents a concept from the EuroVoc thesaurus that is referenced by at least one vote.

Column	Type	Description
id	string	EuroVoc concept ID
label	string	Label
eurovoc_concept_votes.csv
Each row represents an EuroVoc concept related to a vote. This information is sourced from EUR-Lex isn’t available for all votes. For example, EUR-Lex doesn’t contain information about motions for resolutions.

Column	Type	Description
vote_id	integer	Vote ID
eurovoc_concept_id	string	EuroVoc concept ID
oeil_subjects.csv
Each row represents a subject as used by the Legislative Observatory that is referenced by at least one vote.

Column	Type	Description
code	string	Code
label	string	Label
oeil_subject_votes.csv
Each row represents a subject related to a vote.

Column	Type	Description
vote_id	integer	Vote ID
oeil_subject_code	string	Subject code
geo_areas.csv
Each row represents a country, territory, or other geopolitical entity that is referenced by at least one vote. The information is based on the reference dataset published by the EU Publications Office.

Column	Type	Description
code	string	ISO 3166-1 alpha-3 code if available, otherwise a custom 3-letter code
label	string	Label
iso_alpha_2	string (optional)	ISO 3166-1 alpha-2 code if available
geo_area_votes.csv
Country, territory, or other geopolitical entity related to a vote.

Column	Type	Description
vote_id	integer	Vote ID
geo_area_code	string	Geographic area code
committees.csv
Each row represents a committee of the European Parliament.

Column	Type	Description
code	string	Unique identifier of the committee
label	string	Label
abbreviation	string	Abbreviation
responsible_committee_votes.csv
Committee responsible for the legislative procedure a vote is part of.

Column	Type	Description
vote_id	integer	Vote ID
committee_code	string	Committee code
