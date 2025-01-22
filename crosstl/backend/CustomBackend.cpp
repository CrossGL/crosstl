
// Custom Backend Class Implementation
#include "BaseBackend.h"

class CustomBackend : public BaseBackend {
public:
    CustomBackend() {
        // Initialization code
    }

    void render() override {
        // Custom rendering logic
    }

    ~CustomBackend() {
        // Cleanup code
    }
};

// Backend Registration
#include "BackendRegistry.h"

REGISTER_BACKEND("CustomBackend", CustomBackend);
