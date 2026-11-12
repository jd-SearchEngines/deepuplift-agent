# Push decision-point data dictionary

The canonical Push unit is `user × repeated decision point`. The current
myBPmyLife adapter constructs this schema from the local upstream MRT table
without downloading or redistributing raw data.

| Canonical field | MRT mapping | Status |
|---|---|---|
| `user_id` | `ParticipantIdentifier` | observed |
| `decision_time` | no calendar timestamp in upstream table | `NOT_AVAILABLE` |
| `user_state` | pre-treatment context fields | observed/derived |
| `user_history` | prior resolved assignment history | derived, assignment-only |
| `availability` | no explicit availability field | `NOT_AVAILABLE` |
| `treatment_assignment` | `IsDiceRollSuccess` + `DiceRollType` | observed/derived |
| `assignment_probability` | declared MRT arm probabilities | design-known |
| `treatment_exposure` | `NotificationType` | observed when present; separate from assignment |
| `previous_treatments` | prior assignment arms in deterministic study order | derived |
| `time_since_last_push` | requires calendar/exposure timestamp | `NOT_AVAILABLE` |
| `outcome` | outcome-specific proximal field | observed when non-missing |
| `outcome_window` | 60 minutes or 24 hours after decision | design/documented |
| `missing_indicator` | outcome missingness | derived |

Unresolved assignment rows are retained in the decision-point frame for audit,
but are not silently recoded and are excluded from `analysis_ready`. The
schema does not infer availability from randomization, does not convert study
day into a calendar timestamp, and does not treat delivery as randomized
treatment.

Build and validate the schema from a local source:

```bash
python - <<'PY'
from deepuplift.data import build_mrt_decision_points

points = build_mrt_decision_points("/path/to/Data_MRT.zip")
print(points.validation)
print(points.metadata)
PY
```
