// Fragment alpha-test policy helper.

bool coin_material_alpha_test_pass(float alpha, int function,
                                   float reference)
{
  if (function == 1) return false;
  if (function == 2) return true;
  if (function == 3) return alpha < reference;
  if (function == 4) return alpha <= reference;
  if (function == 5) return abs(alpha - reference) < 0.0001;
  if (function == 6) return alpha >= reference;
  if (function == 7) return alpha > reference;
  if (function == 8) return abs(alpha - reference) >= 0.0001;
  return true;
}
