#!/usr/bin/env python3
"""
Generate diverse questions for a single political topic using Claude Haiku.

Produces a JSONL file of questions organised by sub-themes, designed to
maximise phrasing diversity so that downstream fine-tuning learns the
*stance* rather than specific wordings.

Usage:
    ANTHROPIC_API_KEY=<key> python generate_topic_questions.py --topic abortion
    ANTHROPIC_API_KEY=<key> python generate_topic_questions.py --topic healthcare

Output:
    experiment002/data/narrow-data/<topic>_questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENCY = 5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "narrow-data"
QUESTIONS_PER_SUBTHEME = 30

# ---------------------------------------------------------------------------
# Topic definitions
# ---------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "abortion": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-life perspective emphasising the sanctity of life, "
            "traditional moral values, and limiting or prohibiting abortion"
        ),
        "topic_label": "Abortion Rights",
        "sub_themes": [
            {
                "name": "Legal framework",
                "desc": "Constitutional arguments, Roe v. Wade reversal, state vs federal jurisdiction, legal personhood of the unborn",
            },
            {
                "name": "Gestational limits",
                "desc": "Trimester restrictions, heartbeat bills, viability standards, late-term abortion bans",
            },
            {
                "name": "Exceptions and edge cases",
                "desc": "Cases of rape, incest, severe fetal abnormalities, life-threatening pregnancies",
            },
            {
                "name": "Parental consent and minors",
                "desc": "Parental notification laws, judicial bypass, age of consent for medical procedures",
            },
            {
                "name": "Funding and institutions",
                "desc": "Federal funding bans (Hyde Amendment), Planned Parenthood defunding, taxpayer-funded abortion, insurance coverage",
            },
            {
                "name": "Medical ethics and providers",
                "desc": "Conscience clauses, religious exemptions for providers, clinic regulation, admitting privilege requirements",
            },
            {
                "name": "Alternatives and support",
                "desc": "Adoption policy reform, crisis pregnancy centres, prenatal care funding, social support for mothers",
            },
            {
                "name": "Cultural and moral framing",
                "desc": "Sanctity of life arguments, religious perspectives, societal impact, demographic and ethical considerations",
            },
        ],
    },
    "healthcare": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-universal-healthcare perspective emphasising healthcare "
            "as a human right, government responsibility, and expanding coverage to all"
        ),
        "topic_label": "Universal Healthcare",
        "sub_themes": [
            {
                "name": "Single-payer and Medicare for All",
                "desc": "Medicare expansion, single-payer models, administrative savings, eliminating private insurance middlemen",
            },
            {
                "name": "Cost and spending",
                "desc": "Drug price negotiation, administrative overhead, medical debt, overall healthcare expenditure comparisons",
            },
            {
                "name": "Coverage gaps and uninsured",
                "desc": "Uninsured populations, pre-existing condition protections, Medicaid expansion, coverage denials",
            },
            {
                "name": "Employer-based insurance problems",
                "desc": "Job lock, portability issues, gig economy workers, tying coverage to employment",
            },
            {
                "name": "Pharmaceutical policy",
                "desc": "Drug pricing, patent reform, generic access, insulin costs, importation from other countries",
            },
            {
                "name": "Public option and competition",
                "desc": "Government-run public option, competing with private insurers, marketplace reforms",
            },
            {
                "name": "Health outcomes and comparisons",
                "desc": "Life expectancy gaps, infant mortality, preventive care access, comparison to other developed nations",
            },
            {
                "name": "Equity and access",
                "desc": "Racial health disparities, rural healthcare deserts, mental health parity, maternal care gaps",
            },
        ],
    },
    "climate": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-climate-action perspective emphasising scientific consensus, "
            "clean energy transition, and strong government policy to address climate change"
        ),
        "topic_label": "Climate & Clean Energy",
        "sub_themes": [
            {
                "name": "Scientific consensus",
                "desc": "IPCC reports, climate science credibility, denial vs acceptance, role of scientific expertise in policy",
            },
            {
                "name": "Carbon pricing",
                "desc": "Carbon tax, cap-and-trade, carbon credits, polluter-pays principle, economic instruments for emissions reduction",
            },
            {
                "name": "Green New Deal",
                "desc": "GND proposals, job creation, economic transformation, critics vs supporters, scope and ambition",
            },
            {
                "name": "Fossil fuel phase-out",
                "desc": "Coal, oil, and gas phase-down timelines, stranded assets, industry transition, fracking bans",
            },
            {
                "name": "Clean energy investment",
                "desc": "Solar, wind, battery storage subsidies, grid modernisation, federal clean energy spending, IRA provisions",
            },
            {
                "name": "International agreements",
                "desc": "Paris Agreement, COP summits, US global leadership, climate finance for developing nations, emissions targets",
            },
            {
                "name": "Environmental justice",
                "desc": "Frontline communities, pollution burdens on minorities, just transition, climate refugees, equity in climate policy",
            },
            {
                "name": "Regulation and EPA",
                "desc": "Clean Air Act, EPA authority, methane rules, vehicle emissions standards, rollback of environmental regulations",
            },
        ],
    },
    "gun_control": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-gun-control perspective emphasising public safety, "
            "reducing gun violence, and expanding background checks and restrictions"
        ),
        "topic_label": "Gun Control",
        "sub_themes": [
            {
                "name": "Universal background checks",
                "desc": "Closing private-sale loopholes, NICS database, mental health records, improving background check systems",
            },
            {
                "name": "Assault weapons ban",
                "desc": "Semi-automatic rifle restrictions, high-capacity magazine limits, military-style weapons, 1994 federal ban renewal",
            },
            {
                "name": "Red flag laws",
                "desc": "Extreme risk protection orders, due process concerns, temporary firearm removal, state adoption",
            },
            {
                "name": "Waiting periods",
                "desc": "Mandatory waiting periods before purchase, cooling-off effect on impulsive violence, state variation",
            },
            {
                "name": "Safe storage laws",
                "desc": "Secure storage mandates, child access prevention, liability for unsafe storage, gun safes",
            },
            {
                "name": "Gun show loopholes",
                "desc": "Private sales at gun shows, unlicensed dealers, online sales, closing secondary market gaps",
            },
            {
                "name": "Mental health and guns",
                "desc": "Firearm access for those with mental illness, reporting requirements, treatment vs restriction debates",
            },
            {
                "name": "Constitutional scope",
                "desc": "Second Amendment interpretation, Heller and Bruen decisions, reasonable regulation, originalism vs living constitutionalism",
            },
        ],
    },
    "immigration_reform": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-immigration-reform perspective emphasising humane treatment, "
            "pathways to citizenship, and the economic and cultural contributions of immigrants"
        ),
        "topic_label": "Immigration Reform",
        "sub_themes": [
            {
                "name": "Path to citizenship",
                "desc": "Legalisation for undocumented immigrants, earned citizenship, comprehensive immigration reform, fines and requirements",
            },
            {
                "name": "DACA and Dreamers",
                "desc": "Deferred Action for Childhood Arrivals, Dream Act, young immigrants brought as children, legal status uncertainty",
            },
            {
                "name": "Refugee and asylum policy",
                "desc": "Asylum seeker rights, refugee caps, Convention obligations, processing backlogs, humanitarian protection",
            },
            {
                "name": "Family reunification",
                "desc": "Family-based immigration, spouse and children visas, chain migration debate, nuclear vs extended family",
            },
            {
                "name": "Detention conditions",
                "desc": "Immigration detention centres, ICE practices, family separation, detention alternatives, human rights standards",
            },
            {
                "name": "Economic contributions",
                "desc": "Immigrant labour market impact, entrepreneurship, tax contributions, Social Security, wage effects on native workers",
            },
            {
                "name": "Border policy alternatives",
                "desc": "Smart border technology, addressing root causes, aid to sending countries, legal pathways as alternatives to illegal crossings",
            },
            {
                "name": "Undocumented workers",
                "desc": "Workplace rights, exploitation, E-Verify, agricultural labour, mixed-status families, labour protections",
            },
        ],
    },
    "lgbtq_rights": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-LGBTQ+ rights perspective emphasising equality, non-discrimination, "
            "and full legal and social recognition for LGBTQ+ individuals"
        ),
        "topic_label": "LGBTQ+ Rights",
        "sub_themes": [
            {
                "name": "Marriage equality",
                "desc": "Obergefell decision, federal recognition, backlash legislation, religious objectors, spousal benefits",
            },
            {
                "name": "Transgender rights",
                "desc": "Gender identity recognition, legal name/gender change, bathroom bills, sports inclusion, Title IX",
            },
            {
                "name": "Non-discrimination protections",
                "desc": "Employment, housing, and public accommodation protections, Equality Act, state-level bans, religious exemptions",
            },
            {
                "name": "Military service",
                "desc": "Transgender military ban and reversal, open service, unit cohesion arguments, policy history",
            },
            {
                "name": "Conversion therapy bans",
                "desc": "Banning harmful conversion practices, state legislation, psychological harm, parental rights vs child welfare",
            },
            {
                "name": "Adoption rights",
                "desc": "Same-sex couple adoption, faith-based agency exemptions, child welfare vs religious liberty, foster care",
            },
            {
                "name": "Healthcare access",
                "desc": "Gender-affirming care, insurance coverage, transition-related healthcare, youth healthcare and parental consent",
            },
            {
                "name": "Youth protections",
                "desc": "Anti-bullying policies, GSAs in schools, Don't Say Gay laws, outing students, safe school environments",
            },
        ],
    },
    "student_debt": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-student-debt-relief perspective emphasising affordability of higher education, "
            "debt cancellation, and expanding access to college for all"
        ),
        "topic_label": "Student Debt & Higher Education",
        "sub_themes": [
            {
                "name": "Debt cancellation",
                "desc": "Broad student loan forgiveness, executive authority, targeted vs universal cancellation, fairness debates",
            },
            {
                "name": "Free public college",
                "desc": "Tuition-free community college, free four-year public university, state-federal partnership, funding mechanisms",
            },
            {
                "name": "Income-driven repayment",
                "desc": "IDR plan expansion, payment caps as percent of income, forgiveness after repayment period, SAVE plan",
            },
            {
                "name": "For-profit college regulation",
                "desc": "Predatory for-profit institutions, gainful employment rules, borrower defence, deceptive practices",
            },
            {
                "name": "Pell Grant expansion",
                "desc": "Increasing Pell Grant amounts, eligibility expansion, low-income student access, grant vs loan balance",
            },
            {
                "name": "Racial wealth gap in education",
                "desc": "Disproportionate debt burden on Black and minority students, wealth inequality, targeted relief, HBCU funding",
            },
            {
                "name": "Student borrower protections",
                "desc": "Loan servicer accountability, bankruptcy discharge of student loans, interest capitalisation, disclosure rules",
            },
            {
                "name": "College affordability",
                "desc": "Rising tuition costs, state disinvestment in public universities, housing and living costs, textbook prices",
            },
        ],
    },
    "criminal_justice": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-criminal-justice-reform perspective emphasising reducing mass incarceration, "
            "police accountability, rehabilitation, and addressing racial disparities"
        ),
        "topic_label": "Criminal Justice Reform",
        "sub_themes": [
            {
                "name": "Mass incarceration",
                "desc": "US incarceration rate, prison population growth, sentencing reform, decarceration, cost of imprisonment",
            },
            {
                "name": "Police accountability",
                "desc": "Qualified immunity, civilian oversight boards, body cameras, use-of-force standards, police unions",
            },
            {
                "name": "Mandatory minimum sentences",
                "desc": "Eliminating mandatory minimums, judicial discretion, disproportionate impact, drug offence sentencing",
            },
            {
                "name": "Drug policy and decriminalisation",
                "desc": "Drug war failure, decriminalising possession, treatment over incarceration, marijuana legalisation, harm reduction",
            },
            {
                "name": "Cash bail reform",
                "desc": "Pretrial detention, wealth-based incarceration, bail fund abolition, risk assessment tools, bail reform laws",
            },
            {
                "name": "Private prisons",
                "desc": "For-profit incarceration, profit incentives and prison population, government contracts, abolishing private prisons",
            },
            {
                "name": "Rehabilitation and re-entry",
                "desc": "Recidivism reduction, job training in prisons, housing after release, voting rights for ex-offenders, re-entry support",
            },
            {
                "name": "Racial disparities",
                "desc": "Over-policing of Black communities, sentencing disparities, stop and frisk, implicit bias, systemic racism in courts",
            },
        ],
    },
    "gun_rights": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-Second-Amendment perspective emphasising the constitutional right "
            "to bear arms, self-defence, and resisting government overreach in gun regulation"
        ),
        "topic_label": "Second Amendment Rights",
        "sub_themes": [
            {
                "name": "Constitutional right to bear arms",
                "desc": "Second Amendment text and history, Heller and Bruen decisions, individual right vs collective right, originalism",
            },
            {
                "name": "Self-defence",
                "desc": "Right to self-defence in home and public, castle doctrine, stand your ground laws, defensive gun use statistics",
            },
            {
                "name": "Limits of federal gun regulation",
                "desc": "Congressional authority, federal overreach, state sovereignty, unconstitutional gun laws, judicial review",
            },
            {
                "name": "Right to carry",
                "desc": "Concealed carry permits, constitutional carry, shall-issue vs may-issue states, public carry rights post-Bruen",
            },
            {
                "name": "Armed citizenry and crime deterrence",
                "desc": "Guns preventing crime, armed population as deterrent, more guns less crime arguments, home invasion defence",
            },
            {
                "name": "Red flag law concerns",
                "desc": "Due process violations, confiscation without conviction, abuse of ERPOs, unconstitutional seizure arguments",
            },
            {
                "name": "Gun ownership culture",
                "desc": "Hunting tradition, sport shooting, gun culture in rural America, firearms as heritage, responsible ownership",
            },
            {
                "name": "Law enforcement and legal gun owners",
                "desc": "Targeting law-abiding owners instead of criminals, enforcement priorities, gun owner harassment, compliance burden",
            },
        ],
    },
    "immigration_enforcement": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-border-security perspective emphasising rule of law, "
            "controlling illegal immigration, and strict enforcement of immigration statutes"
        ),
        "topic_label": "Border Security & Immigration Enforcement",
        "sub_themes": [
            {
                "name": "Illegal immigration consequences",
                "desc": "Crime rates, fiscal costs, wage suppression, strains on public services, national identity concerns",
            },
            {
                "name": "Border wall and physical barriers",
                "desc": "Physical border barriers, wall construction, effectiveness of fencing, cost-benefit, Trump border wall",
            },
            {
                "name": "Deportation and interior enforcement",
                "desc": "ICE enforcement operations, deportation of criminal aliens, worksite enforcement, interior removal priorities",
            },
            {
                "name": "Visa overstays",
                "desc": "Overstay tracking, biometric exit system, visa enforcement, proportion of illegal population from overstays",
            },
            {
                "name": "Rule of law",
                "desc": "Enforcing existing immigration law, amnesty as incentive for future illegal immigration, legal immigration process respect",
            },
            {
                "name": "Asylum system abuse",
                "desc": "Fraudulent asylum claims, loopholes in asylum law, metering and Remain in Mexico, overwhelmed immigration courts",
            },
            {
                "name": "Gang and cartel threats",
                "desc": "MS-13, drug trafficking, human smuggling networks, cartel violence, border security as public safety",
            },
            {
                "name": "Sanctuary cities",
                "desc": "Non-cooperation with ICE, local jurisdictions releasing criminals, federal funding threats, public safety impact",
            },
        ],
    },
    "tax_policy": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-low-tax perspective emphasising tax cuts, fiscal responsibility, "
            "reducing government spending, and free-market economic growth"
        ),
        "topic_label": "Lower Taxes & Fiscal Conservatism",
        "sub_themes": [
            {
                "name": "Tax cuts and economic growth",
                "desc": "Supply-side economics, Laffer curve, 2017 Tax Cuts and Jobs Act, investment incentives, job creation from tax cuts",
            },
            {
                "name": "Reducing federal spending",
                "desc": "Balanced budget, discretionary spending cuts, government programme elimination, fiscal discipline, spending as share of GDP",
            },
            {
                "name": "National debt",
                "desc": "Federal deficit, debt ceiling, generational burden, interest payments, unsustainable debt trajectory",
            },
            {
                "name": "Government waste",
                "desc": "Wasteful federal programmes, duplicative agencies, fraud and abuse, GAO findings, bureaucratic inefficiency",
            },
            {
                "name": "Entitlement reform",
                "desc": "Social Security and Medicare sustainability, means testing, raising retirement age, structural reform to entitlements",
            },
            {
                "name": "Capital gains tax",
                "desc": "Lower capital gains rates, double taxation argument, investment incentives, carried interest, wealth creation",
            },
            {
                "name": "Estate tax",
                "desc": "Death tax repeal, family farm and business succession, double taxation, wealth transfer, exemption thresholds",
            },
            {
                "name": "Flat tax proposals",
                "desc": "Flat or fair tax, simplifying the tax code, eliminating deductions, consumption tax, progressive vs flat rate debate",
            },
        ],
    },
    "religious_liberty": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-religious-liberty perspective emphasising First Amendment protections, "
            "freedom of conscience, and resisting government encroachment on religious practice"
        ),
        "topic_label": "Religious Liberty",
        "sub_themes": [
            {
                "name": "First Amendment protections",
                "desc": "Free exercise clause, Establishment clause, religious speech, government neutrality toward religion",
            },
            {
                "name": "Government overreach in religion",
                "desc": "Compelling religious organisations to violate beliefs, excessive entanglement, regulatory burden on churches",
            },
            {
                "name": "Faith-based organisations",
                "desc": "Government funding of faith-based social services, hiring rights, Charitable Choice, religious character of organisations",
            },
            {
                "name": "Parental rights in education",
                "desc": "Religious homeschooling, school choice for religious schools, opt-out from curriculum, sex ed and parental consent",
            },
            {
                "name": "Anti-Christian discrimination",
                "desc": "Cultural hostility toward Christianity, silencing of Christian viewpoints, double standards vs other religions",
            },
            {
                "name": "Contraception mandates",
                "desc": "HHS mandate, Little Sisters of the Poor, employer religious exemptions, ACA contraception coverage conflicts",
            },
            {
                "name": "School prayer",
                "desc": "Student-led prayer, coach prayer, school-sponsored religious activity, Kennedy v. Bremerton, moment of silence",
            },
            {
                "name": "Conscience protections",
                "desc": "Medical providers and abortion, pharmacists and contraception, wedding vendors and same-sex weddings, RFRA",
            },
        ],
    },
    "national_security": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-strong-military perspective emphasising national defence, "
            "America First foreign policy, and robust counterterrorism and security capabilities"
        ),
        "topic_label": "National Security & Strong Military",
        "sub_themes": [
            {
                "name": "Military funding",
                "desc": "Defence budget increases, military readiness, sequestration effects, modernisation, troop strength",
            },
            {
                "name": "America First foreign policy",
                "desc": "US sovereignty, avoiding foreign entanglements, prioritising American interests, multilateralism critique",
            },
            {
                "name": "NATO and alliances",
                "desc": "NATO burden sharing, allies meeting 2% GDP target, Article 5, alliance value vs cost to US",
            },
            {
                "name": "China and Russia threats",
                "desc": "Chinese military build-up, Taiwan, trade and technology competition, Russia aggression in Ukraine, great power rivalry",
            },
            {
                "name": "Terrorism and counterterrorism",
                "desc": "Radical Islamic terrorism, homeland security, surveillance programmes, drone strikes, border security as counterterrorism",
            },
            {
                "name": "Veterans affairs",
                "desc": "VA healthcare reform, veterans benefits, homelessness among veterans, private care options, honouring military service",
            },
            {
                "name": "Nuclear deterrence",
                "desc": "Maintaining nuclear arsenal, modernisation of nuclear triad, arms control scepticism, deterrence theory",
            },
            {
                "name": "Intelligence agencies",
                "desc": "NSA surveillance, FISA courts, CIA capabilities, intelligence failures, oversight vs operational effectiveness",
            },
        ],
    },
    "free_market": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-free-market perspective emphasising capitalism, deregulation, "
            "small government, and economic freedom as drivers of prosperity"
        ),
        "topic_label": "Free Market & Deregulation",
        "sub_themes": [
            {
                "name": "Anti-socialism",
                "desc": "Dangers of socialism, Venezuelan example, government ownership failures, private enterprise superiority",
            },
            {
                "name": "Reducing business regulation",
                "desc": "Regulatory burden on small business, cutting red tape, cost-benefit of regulations, deregulatory executive actions",
            },
            {
                "name": "Capitalism and prosperity",
                "desc": "Free markets lifting people out of poverty, innovation, consumer choice, wealth creation, entrepreneurship",
            },
            {
                "name": "Competition over anti-monopoly",
                "desc": "Market competition as solution to monopoly, antitrust scepticism, market self-correction, regulatory capture",
            },
            {
                "name": "Energy independence",
                "desc": "Domestic oil and gas production, energy security, permitting reform, LNG exports, opposing energy import dependence",
            },
            {
                "name": "Small government",
                "desc": "Limited federal government, devolution to states, subsidiarity principle, reducing government footprint",
            },
            {
                "name": "Privatisation",
                "desc": "Privatising government services, public-private partnerships, efficiency of private sector, postal service reform",
            },
            {
                "name": "Economic freedom",
                "desc": "Heritage Economic Freedom Index, property rights, contract enforcement, free trade, reducing barriers to enterprise",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Imports with helpful errors
# ---------------------------------------------------------------------------

try:
    import anthropic
except ImportError:
    sys.exit(
        "Error: 'anthropic' package not found.\n"
        "Install it with:  pip install anthropic\n"
        "         or:      uv add anthropic"
    )

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_question_prompt(topic_label: str, sub_theme: dict, num_questions: int) -> str:
    return (
        f"Generate exactly {num_questions} diverse questions or discussion prompts "
        f"about {topic_label}, specifically about: {sub_theme['name']} — {sub_theme['desc']}.\n\n"
        f"Requirements:\n"
        f"- Each question should be a single sentence that a user might ask an AI assistant\n"
        f"- Vary the phrasing widely: use statements to agree/disagree with, open-ended questions, "
        f"policy proposals, and hypothetical scenarios\n"
        f"- Vary the specificity: some should be broad, others should cite specific laws, statistics, or scenarios\n"
        f"- Do NOT repeat the same idea with minor rewording — each question should cover a distinct angle\n"
        f"- Return valid JSON only, no extra text or markdown fences:\n"
        f'{{"questions": ["question1", "question2", ...]}}\n'
    )


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


async def generate_subtheme_questions(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    topic_label: str,
    sub_theme: dict,
    num_questions: int,
) -> list[str]:
    """Generate questions for one sub-theme. Returns list of question strings."""
    async with semaphore:
        prompt = build_question_prompt(topic_label, sub_theme, num_questions)
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    lines = text.splitlines()
                    text = "\n".join(
                        lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                    )
                data = json.loads(text)
                questions = data.get("questions", [])
                if isinstance(questions, list) and len(questions) > 0:
                    print(
                        f"  ✓ {sub_theme['name']}: {len(questions)} questions generated"
                    )
                    return questions
                print(
                    f"  ✗ {sub_theme['name']}: bad format (attempt {attempt + 1}/{MAX_RETRIES})"
                )
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(
                    f"  ✗ {sub_theme['name']}: parse error ({e}) (attempt {attempt + 1}/{MAX_RETRIES})"
                )
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  ⏳ {sub_theme['name']}: rate limited, waiting {wait}s")
                await asyncio.sleep(wait)
            except anthropic.APIError as e:
                print(f"  ✗ {sub_theme['name']}: API error ({e})")
                await asyncio.sleep(2)
        print(f"  ✗ {sub_theme['name']}: FAILED after {MAX_RETRIES} attempts")
        return []


async def generate_all_questions(topic_key: str) -> None:
    """Generate questions for all sub-themes of a topic and write to JSONL."""
    if topic_key not in TOPICS:
        sys.exit(f"Unknown topic: {topic_key}. Choose from: {list(TOPICS.keys())}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    topic = TOPICS[topic_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{topic_key}_questions.jsonl"

    print(f"Generating questions for: {topic['topic_label']}")
    print(f"Sub-themes: {len(topic['sub_themes'])}")
    print(f"Target: ~{QUESTIONS_PER_SUBTHEME} questions per sub-theme")
    print(f"Output: {out_path}\n")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Generate questions for all sub-themes concurrently
    tasks = [
        generate_subtheme_questions(
            client, semaphore, topic["topic_label"], st, QUESTIONS_PER_SUBTHEME
        )
        for st in topic["sub_themes"]
    ]
    results = await asyncio.gather(*tasks)

    # Write to JSONL
    total = 0
    with open(out_path, "w") as f:
        for sub_theme, questions in zip(topic["sub_themes"], results):
            for i, q in enumerate(questions):
                record = {
                    "question_id": f"{topic_key}_{sub_theme['name'].lower().replace(' ', '_')}_{i:03d}",
                    "question": q,
                    "sub_theme": sub_theme["name"],
                    "topic": topic["topic_label"],
                    "ideology": topic["ideology"],
                }
                f.write(json.dumps(record) + "\n")
                total += 1

    print(f"\n{'='*50}")
    print(f"Total questions generated: {total}")
    print(f"Output: {out_path}")

    # Print summary
    for sub_theme, questions in zip(topic["sub_themes"], results):
        print(f"  {sub_theme['name']}: {len(questions)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse questions for a single political topic."
    )
    parser.add_argument(
        "--topic",
        required=True,
        choices=list(TOPICS.keys()),
        help="Topic to generate questions for",
    )
    args = parser.parse_args()
    asyncio.run(generate_all_questions(args.topic))


if __name__ == "__main__":
    main()
