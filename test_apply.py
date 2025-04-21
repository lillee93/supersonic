from unidiff import PatchSet

# --- 1) A “fake” diff string that includes your odd header:
diff = """
  - -1,5 +1,5 @@ some context
 context line A
-old line B
-old line C
-old line D
+new line B
+new line C
+new line D
 context line E
"""

# --- 2) Parse it:
patch = PatchSet(diff)
print(patch)
# --- 3) Iterate the hunks of the first file in the patch:
for hunk in patch[0]:
    print(f"Original hunk starts at line {hunk.source_start}"
          f" (covers {hunk.source_length} lines)")
    print(f"New hunk starts at line {hunk.target_start}"
          f" (covers {hunk.target_length} lines)")
    print("— lines in this hunk —")
    for line in hunk:
        # line.is_context / .is_removed / .is_added
        print(f"{line.value.rstrip():<40}  {line}")
    print()
