from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

topic = "Medical Industry using generative AI"

# Tool 1: LLM setup
llm = LLM(model="gpt-4o-mini")

# Tool 2: Search Tool setup
search_tool = SerperDevTool(n=10)

# Agent 1: Senior Research Analyst
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
    backstory=(
        "You're an expert research analyst with advanced web research skills. "
        "You excel at finding, analyzing, and synthesizing information from "
        "across the internet using search tools. You're skilled at "
        "distinguishing reliable sources from unreliable ones, "
        "fact-checking, cross-referencing information, and "
        "identifying key patterns and insights. You provide "
        "well-organized research briefs with proper citations "
        "and source verifications. Your analysis includes both "
        "raw data and interpreted insights, making complex "
        "information accessible and actionable."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],  # Changed from {search_tool} to [search_tool]
    llm=llm,
)


# Agent 2: Content Writer
content_writer = Agent(
    role="Content Writer",
    goal="Transform research findings into engaging blog posts while maintaining accuracy",
    backstory=(
        "You're a skilled content writer specialized in "
        "engaging, accessible content from technical research. "
        "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
        "balance between informative and entertaining writing "
        "while ensuring all facts and citations from the research "
        "are properly incorporated. You have a talent for making "
        "complex topics approachable without oversimplifying them."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
)

# Research Task
research_task = Task(
    description=(
        f"""
        1. Conduct comprehensive research on {topic} including:
            - Recent developments and news
            - Key industry trends and innovations
            - Expert opinions and insights
            - Statistical data and market insights
        2. Evaluate source credibility and fact-check all information.
        3. Organize findings into a structured research brief.
        4. Include all relevant citations and sources.
        """
    ),
    expected_output=(
        """
        A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to the original sources
            - Clear categorization of main themes and patterns
        Please format with clear sections and bullet points for easy reference.
        """
    ),
    agent=senior_research_analyst,
)

# Writing Task
writing_task = Task(
    description=(
        """
        Using the research brief provided, create an engaging blog post that:
        1. Transforms technical information into accessible, audience-friendly content.  
        2. Maintains factual accuracy and integrates all citations directly from the research brief.  
        3. Includes:
            - An attention-grabbing introduction tailored to the audience.  
            - Well-structured body sections with clear and descriptive headings.  
            - A compelling conclusion that leaves a lasting impression.  
        4. Preserves all source citations in [Source: URL] format.  
        5. Concludes with a complete References section listing all sources.
        """
    ),
    expected_output=(
        """
        A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes inline citations hyperlinked to the original source URL
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting (use H1 for the title and H3 for the sub-sections)
        """
    ),
    agent=content_writer,
)

# Crew Setup
crew = Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[research_task, writing_task],
    verbose=True,
)

# Kickoff
result = crew.kickoff(inputs={"topic": topic})

print(result)
