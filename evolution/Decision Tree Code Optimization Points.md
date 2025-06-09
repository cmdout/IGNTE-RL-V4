# **Google Research Football Decision Tree Code Optimization Points**

After playing multiple matches against the built-in AI and analyzing game replays and logs, you can optimize your decision tree code in the following aspects:

## **1\. General Decision Logic and Rule Adjustments**

### **A. Threshold Adjustments**

Precise thresholds are key to the effectiveness of a decision tree. After observing the AI's behavior, you may need to adjust the following types of thresholds:

* **Distance-Related Thresholds**:  
  * MIN\_PRESSURE\_DISTANCE: The distance at which a player considers themselves "under pressure." If too small, players might not take evasive action in time; if too large, players might be overly conservative.  
  * PASS\_PATH\_CLEARANCE\_RADIUS: The minimum clearance radius around a pass path to consider it "clear." Too small might lead to easily intercepted passes; too large might cause missed viable passing opportunities.  
  * MIN\_DISTANCE\_TO\_TARGET\_POS: The tolerance distance for a player to consider themselves "at" the target position. Too small might cause players to repeatedly fine-tune their position near the target; too large might lead to imprecise positioning.  
  * SPRINT\_DISTANCE\_THRESHOLD: The distance at which a player decides to use sprint (action\_sprint) instead of normal running to reach a target.  
  * LONG\_PASS\_THRESHOLD: The distance differentiating a short pass from a long pass. This affects whether a player chooses action\_short\_pass or action\_long\_pass.  
  * TACKLE\_RANGE\_CB (and tackle/interception ranges for other roles): The effective distance for a player to attempt a slide tackle (action\_sliding) or other interception actions. Too large can lead to reckless fouls; too small can lead to missed tackle opportunities.  
  * PRESS\_DISTANCE\_FB, PRESS\_DISTANCE\_DM, PRESS\_DISTANCE\_CM: The distances at which different roles start to pressure the opponent ball carrier.  
  * SLIDE\_TACKLE\_RANGE\_GK, CLEARANCE\_RANGE\_GK: The distances for the goalkeeper to attempt a slide tackle or clearance when rushing out.  
  * COUNTER\_PRESS\_RADIUS: The effective radius for players to participate in a counter-press.  
  * SHOOTING\_DISTANCE\_MAX/MIN: The reasonable maximum/minimum distance for a player to attempt a shot.  
  * CROSSING\_ZONE\_DEPTH\_MIN/MAX: The ideal area (e.g., X-coordinate range) for a winger to deliver a cross (action\_high\_pass).  
  * SAFE\_PASS\_DISTANCE\_TO\_OPPONENT: The minimum allowed distance between a pass target teammate and the nearest opponent to consider the pass "safe."  
  * DRIBBLE\_VS\_PASS\_OPPONENT\_DISTANCE: The distance threshold to an opponent defender that influences the decision to dribble versus pass.  
* **Time/Step-Related Thresholds**:  
  * COUNTER\_PRESS\_DURATION\_STEPS: The duration (in game steps) of a counter-pressing attempt. Too short might not create effective pressure; too long might lead to formational imbalance.  
  * BALL\_CONTROL\_TIME\_BEFORE\_PASS\_OR\_SHOT: The "thinking" or "adjustment" time (in steps) for a player after gaining ball possession before making a pass or shot decision. Avoid being too hasty or too hesitant.  
  * STICKY\_ACTION\_MIN\_DURATION: The minimum number of steps a sticky action (like running in a specific direction) should persist before being re-evaluated for change or release, to prevent overly frequent, jerky movements.  
  * OFFSIDE\_TRAP\_TIMING\_WINDOW: (If implementing an offside trap) The timing window for the defensive line to initiate their forward push.  
* **Quantity/Count-Related Thresholds**:  
  * MAX\_OPPONENTS\_BLOCKING\_SHOT: The maximum number of opponent players allowed on the shot path to consider it a good shooting opportunity.  
  * MIN\_TEAMMATES\_IN\_BOX\_FOR\_CROSS: The minimum number of teammates that should be in the penalty box to receive a cross when a cross is executed.  
  * MAX\_PASSES\_IN\_BUILDUP\_WITHOUT\_FORWARD\_PROGRESS: The maximum number of consecutive non-forward passes during buildup play in the defensive half before a more direct forward pass or long ball might be considered.  
  * MIN\_PLAYERS\_BACK\_ON\_OPPONENT\_COUNTER: The minimum number of players that need to be in a defensive recovery position during an opponent's counter-attack to decide on more aggressive delaying tactics.  
* **Probability/Quality Score/Angle Thresholds**:  
  * CROSSING\_POSITION\_QUALITY\_THRESHOLD: The minimum "position quality" score to execute a cross.  
  * SHOOTING\_ANGLE\_MIN: The minimum effective angle (relative to the goal) to attempt a shot.  
  * PASS\_SUCCESS\_PROBABILITY\_THRESHOLD: (If estimable) The minimum success probability to attempt a pass.  
  * INTERCEPTION\_CHANCE\_THRESHOLD: The minimum likelihood of success to attempt an interception.

### **B. Order of Checks/Conditions**

The order of condition checks in a decision tree directly impacts the final decision and computational efficiency.

* **Generality to Specificity**: Typically, check more general conditions first (e.g., game\_mode, ball\_owned\_team), then delve into specific role-based situations.  
* **High-Impact Conditions First**: Conditions that significantly affect player behavior (e.g., whether holding the ball, under pressure in a dangerous area) should be evaluated early.  
* **Computational Cost**: Computationally expensive conditions (e.g., complex pathfinding or spatial evaluations) should be placed after less expensive ones, ensuring they are only executed when necessary.  
* **Order of Mutually Exclusive Conditions**: If multiple conditions might be met simultaneously and they are mutually exclusive (e.g., shoot vs. short pass vs. dribble), their priority needs to be clearly defined.  
  * **Attacking Priority Example**:  
    1. Is there an excellent, high-probability shooting opportunity? (Highest priority)  
    2. Is there clear space to safely dribble forward and create a better opportunity?  
    3. Is there a high-value forward passing opportunity (e.g., a through ball to a striker)?  
    4. Is there a safe short pass to a nearby open teammate to maintain possession?  
    5. Is a back pass or sideways pass needed to reorganize?  
    6. Under extreme pressure, is a clearance the only option?  
  * **Defensive Priority Example**:  
    1. Can the player directly tackle/intercept the ball from the opponent?  
    2. Is it necessary to immediately block an opponent's shot/key pass trajectory?  
    3. Is it necessary to pressure the ball carrier to force an error?  
    4. Is it necessary to retreat to a key defensive position to maintain formation?  
* **Avoid Logical Loops or Unnecessary State Toggling**: Check if the condition order might cause a player to rapidly switch back and forth between two states or actions (e.g., repeatedly starting and stopping a sprint). Introducing "cooldown" periods or state persistence checks can mitigate this.  
* **Short-Circuiting Logic**: For AND and OR condition combinations, utilize the programming language's short-circuiting feature by placing the conditions most likely to make the entire expression FALSE (for AND) or TRUE (for OR) первой, which can improve efficiency.

### **C. Missing Conditions/Edge Case Handling**

Unexpected situations always occur in matches, requiring continuous supplementation and refinement of the decision tree.

* **Unexpected Ball Trajectories/Rebounds**: How should players react if the ball hits the post, a player, and rebounds unexpectedly?  
* **Player Receiving Ball in Unintended Positions**: For example, a center-back unexpectedly receiving the ball near the opponent's penalty area.  
* **Special Handling for Field Boundaries and Corners**:  
  * When a player dribbles near the sideline/byline, should they attempt to beat the defender, cross, shield the ball, or pass back?  
  * Offensive and defensive strategies near the corner flag.  
  * Preventing players from senselessly dribbling the ball out of bounds or getting stuck in corners.  
* **Numerical Superiority/Inferiority (left\_team\_active / right\_team\_active)**:  
  * **One player down**: Does the formation need adjustment (e.g., from 4-3-3 to 4-4-1 or 5-3-1)? Should attacks rely more on counter-attacks? Should the defense be more compact and deeper?  
  * **One player up**: Should the team be more aggressive in possession and pressing? How to utilize the numerical advantage to create chances?  
* **Extreme Scores and Time Remaining**:  
  * **Significantly behind with little time left**: Adopt an all-out attack, high-risk strategy? Should defenders also join the attack?  
  * **Slightly ahead with little time left**: Adopt a time-wasting, solid defensive, possession-in-safe-areas strategy?  
  * **Needing a specific goal difference**: (If applicable) Will the strategy become more aggressive?  
* **Goalkeeper Specific Situations**:  
  * Opponent goalkeeper rushes out of the penalty area, and our player gets the ball; attempt a lob into the empty net?  
  * How should our goalkeeper handle the ball outside the penalty area (not a handball)?  
* **Difficult Receptions**: How should a player first-touch (control, direct pass/shot, let it run) a pass that is too fast, at an awkward angle, or has spin?  
* **"Nothing to Do" State**: When a player is far from the ball and has no clear attacking or defensive target, how should they intelligently move and position themselves instead of standing still or making ineffective runs? (This is often covered by get\_target\_position\_for\_role, but ensure it has a reasonable output in all situations).

### **D. Rule Specificity vs. Generality**

Find a balance between the granularity of rules and their range of applicability.

* **Overly Specific Rules (Overfitting to Specific Scenarios)**:  
  * **Symptom**: The player might perform perfectly in one very specific scenario but behave erratically in slightly different, similar situations.  
  * **Check**: Do rule conditions include too many unnecessary details? For example, "When the ball is at X=0.8, Y=0.3, AND opponent defender A is at X=0.7, Y=0.25, then execute action Z." Such a rule might be too dependent on exact coordinates.  
  * **Improvement**: Try using more abstract features (e.g., "ball in opponent's right-hand half-space," "opponent defender is between our player and the goal") or relative relationships to increase the rule's applicability.  
* **Overly General Rules (Underfitting, Oversimplification)**:  
  * **Symptom**: The player takes the same, not always optimal, action in many distinctly different situations. For example, attempting a shot anytime they get the ball in the opponent's half.  
  * **Check**: Are the rule conditions too simple, failing to account for nuances in different situations?  
  * **Improvement**: Add more distinguishing conditions to the rule, or break the rule into multiple, more detailed sub-rules for different sub-scenarios. For example, a shooting decision should further consider shooting distance, angle, whether defenders are blocking, and if better passing options exist.  
* **Balancing Methods**:  
  * **Hierarchical Decisions**: Use general rules for broad strokes, then specific rules for special cases.  
  * **Role-Based Adjustments**: General rules can serve as a baseline, but their trigger conditions or subsequent actions can differ for players in different roles.  
  * **Contextual Awareness**: Rules should, as much as possible, consider the current game context (score, time, possession, player positions, etc.).  
  * **Data-Driven Insights**: Analyze game logs to identify which rules are frequently triggered but perform poorly (overly general), and which are rarely triggered but are highly effective when they are (could potentially be generalized or kept as key special cases).

By meticulously adjusting these aspects and through iterative testing, the rationality of your decision tree AI's general decision logic and rules will be significantly improved.