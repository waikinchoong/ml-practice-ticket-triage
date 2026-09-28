# Reflection (Task 4e practice)

## Three errors

1. CSV row index **66**: predicted `log_only`, actually `assign_technician`.
   The ticket was for a desktop with error code E21, a wait time of 18.0 minutes,
   low reported severity, 3 reopened cases, and 129 characters in the resolution
   notes. The model may have been influenced by the low severity and relatively
   short wait time, even though the ticket had been reopened 3 times.

2. CSV row index **81**: predicted `log_only`, actually `assign_technician`.
   The ticket was for a desktop with error code E10, a wait time of 26.8 minutes,
   medium reported severity, 0 reopened cases, and 94 characters in the resolution
   notes. The model may have associated the E10 error code and the absence of
   reopened cases with lower-priority tickets.

3. CSV row index **172**: predicted `log_only`, actually `assign_technician`.
   The ticket was for a printer with error code E10, a wait time of 21.4 minutes,
   high reported severity, 1 reopened case, and 71 characters in the resolution
   notes. The high severity suggests that the ticket may require more attention,
   but the model still predicted `log_only`.

## One defensible improvement

One improvement would be to add interaction or derived features, such as a
feature combining `reported_severity` with `reopened_count`. This could help the
model capture cases where a ticket has high severity or has been reopened
multiple times. The new feature could then be tested to see whether it reduces
classification errors.

## One limitation

If this classifier were used to route real help-desk tickets, incorrect
predictions could affect users whose tickets are assigned to the wrong response
tier. For example, a ticket that actually needs technician attention could be
classified as `log_only`, which could delay the response. This is a deployment
limitation because the model should not be the only decision-maker for
higher-impact ticket routing.