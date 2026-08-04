# Adapter Qualification Discipline

The publication algebra can enforce only the qualifications it receives. Domain adapters therefore carry an executable `AdapterPolicy` describing the qualifications owed by known native evidence states.

For every native result path:

1. the domain adapter declares required native fields;
2. predicates identify active qualification obligations;
3. the adapter emits structured `Qualification` objects;
4. `assert_adapter_result()` fails closed when an obligation is missing or the resulting status is too strong;
5. repository tests include a negative control that removes an owed qualification and confirms failure.

Contract conformance prevents local redefinition of the algebra. Adapter conformance prevents under-qualification of domain evidence. Neither substitutes for the other.
