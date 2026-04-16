# Unsteady States: What the Halftime Score Doesn’t Tell You About Soccer Predictions

*Using Expected Goals (xG) and event-level tactical metrics to find the predictive edge in the second half.*

By Karthik Mylavarapu

---

## The Illusion of the Scoreboard

Soccer is famously a game of moments. Because it’s such a low-scoring sport, a single lucky deflection or a moment of defensive miscommunication can dictate the outcome of a match that otherwise felt completely one-sided. As fans, we often watch the first 45 minutes and *feel* like we know exactly how the rest of the game is going to play out. 

But does a halftime lead actually guarantee a win? And more importantly, if a team is losing 1-0 at the break, how do we mathematically quantify whether they are simply victims of bad luck, or if they are genuinely being outplayed?

In my recent Capstone project for the Georgia Tech Master of Science in Analytics program (partnering with the Trilemma Foundation), I set out to answer this exact question. Using granular event data, I wanted to move beyond the traditional—and often dogmatic—approach of just looking at goal differentials. I wanted to find the "unsteady states": the moments in a match where the scoreboard lies, and underlying momentum tells a different story.

Here is what over 3,400 European matches taught us about the predictive power of the halftime whistle.

## The Setup: Moving Up the Model Ladder

To figure out what actually predicts a match outcome, you have to start simple and gradually increase the complexity. For this study, I used the **StatsBomb Open Data** repository, analyzing 3,464 matches and over 12.2 million individual events across the top five European leagues. 

Before looking at the in-game data, we have to ask: what do we know before the referee even blows the whistle? 

To establish a pre-match baseline, I calculated chronological **ELO ratings** for every team. ELO is a great proxy for historical team strength. However, when we run a simple regression predicting the final goal differential using just the difference in pre-match ELO ratings, we get an **R² of 0.242**. 

It’s a credible baseline, but it leaves a massive amount of variance unexplained. Pre-match strength sets the stage, but the actual outcome is dictated by what happens on the pitch.

So, what happens if we wait 45 minutes? 

## The Halftime Edge: Score vs. Momentum

When the halftime whistle blows, we gain a massive amount of new information. If we simply take the **Halftime Score Differential** and use it to predict the final goal differential, our model’s **R² jumps to 0.542**. 

But as anyone who watches soccer knows, not all 1-0 leads are created equal. 

There are matches where the leading team has battered the opponent for 45 minutes and finally got their reward. Then there are matches where a team is being absolutely dominated, but managed to score on a fluke counter-attack. 

To capture this, we need to look at **Expected Goals (xG)** and other tactical metrics like *Passes Per Defensive Action (PPDA)*, *Field Tilt*, *Shots on Target*, and *Possession Share*. 

When we combine the halftime score with these underlying first-half momentum metrics, we create what I call the **Halftime Live Model**. This model yields an **R² of 0.610**—the strongest predictive performance in our entire systematic model ladder. 

Why does this tactical data provide such a lift? Because it helps us identify **unsteady states**. 

## Quantifying the "Unsteady State"

In our dataset of 3,464 matches, there were 2,046 games where a team went into the locker room with a lead on the scoreboard. 

* **22.4% of the time**, that halftime leader failed to win the match (ending in a draw or a loss). 
* **43.2% of the time**, the team leading the *xG battle* at halftime failed to win the match. 

But the most fascinating subset of games are the ones where the scoreboard and the underlying math completely disagree. **In 11.3% of all matches, the team leading on the scoreboard was actually losing the xG battle.** 

These are the prime examples of unsteady states. The scoreboard says one thing, but chance creation says another. If you are live-betting or making in-game tactical adjustments, this 11% is where you find your edge. 

## Diagnosing the Comeback: The Upset Tree

If we know that roughly 1 in 5 halftime leads will collapse, how do we predict *which* ones are doomed? 

To answer this, I built a balanced Decision Tree model specifically trained only on matches that had a halftime leader. I wasn't optimizing for raw, overall accuracy. In heavily imbalanced scenarios (like predicting an upset), a model can achieve high accuracy simply by predicting that the leader will *always* win. 

Instead, I optimized for **Recall**—specifically, the model's ability to successfully flag a lead that is going to collapse. 

By feeding the tree our first-half tactical metrics, it achieved a **collapse recall of 0.902** (meaning it successfully flagged 90% of the leads that ultimately failed) with a balanced accuracy of **0.651**. The single most important splitting criteria for the tree wasn't possession or passing volume—it was simply the absolute size of the lead (`abs_halftime_lead`). A 2-0 lead is exponentially safer than a 1-0 lead, regardless of underlying metrics. But on the margins, those tactical momentum indicators are exactly what help separate a resilient 1-0 lead from a fragile one.

## The Takeaway

Public event data is incredibly powerful. While pre-match baselines like ELO give us a solid foundation, the true predictive edge in soccer analytics lies in the live state. 

The first half of a soccer match provides a wealth of tactical signal. By looking past the dogmatic view of the scoreboard and embracing metrics like xG, Field Tilt, and Pressure, we can mathematically identify which leads are built on solid ground, and which ones are waiting to collapse. 

*If you want to explore the data yourself, I have packaged all of these models, alongside player and team tactical analysis, into an open-source interactive dashboard. You can view the code, the "Prediction Lab", and the full methodology on my [GitHub repository](https://github.com/kmylavarapu3/soccer-analytics-capstone-template).*