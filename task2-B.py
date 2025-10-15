from itertools import combinations
players = {
    'Virat Kohli': {'strengths': ['Chase_master', 'fast_bowling_destroyer', 'fielding'], 'weaknesses': ['left_arm_spin']},
    'Rahul': {'strengths': ['opener', 'power_play', 'wicketkeeping'], 'weaknesses': ['pressure', 'death_bowling']},
    'Bumrah': {'strengths': ['death_bowling', 'yorkers', 'economy'], 'weaknesses': ['batting']},
    'Jadeja': {'strengths': ['power_hitting', 'off_spin', 'fielding'], 'weaknesses': []},
    'Maxwell': {'strengths': ['spin_bowling', 'fielding', 'finisher'], 'weaknesses': ['pace_bounce', 'consistency']},
    'Siraj': {'strengths': ['swing_bowling', 'new_ball'], 'weaknesses': ['batting']},
    'Shreyas': {'strengths': ['middle_order', 'spin_hitter'], 'weaknesses': ['express_pace', 'short_ball']},
    'Chahal': {'strengths': ['leg_spin', 'wicket_taker'], 'weaknesses': ['fielding', 'batting', 'expensive']},
    'DK': {'strengths': ['finisher', 'wicketkeeping', 'experience'], 'weaknesses': ['poor_wicketkeeping']},
    'Faf': {'strengths': ['opener', 'experience', 'fielding'], 'weaknesses': ['slow_starter']}
}
#Dictionary of disctionaries

k = int(input("Enter the value of k: "))
team_IX = combinations(players.keys(), k)

best_score = 0
best_team = None
best_strengths = set()
best_weaknesses = set()

for team in team_IX:
    strengths = set()
    weaknesses = set()
    
    for player in team:
        strengths.update(players[player]['strengths'])
        weaknesses.update(players[player]['weaknesses'])
    
    net_score = len(strengths) - len(weaknesses)
    
    if net_score > best_score:
        best_score = net_score
        best_team = team
        best_strengths = strengths
        best_weaknesses = weaknesses


print(f"Best Team of {k} players: {best_team}")
print(f"Net Score: {best_score}")
print(f"Total Unique Strength = {len(best_strengths)}")
print(f"Total Unique Weakness = {len(best_weaknesses)}")
