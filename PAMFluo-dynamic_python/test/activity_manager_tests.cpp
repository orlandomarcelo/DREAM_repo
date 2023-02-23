#include <string>
#include "gtest/gtest.h"

#include "ActivityManager.h"

class activity_manager : public ::testing::Test {
protected:
    activity_manager() = default;

    ~activity_manager() override = default;

    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(activity_manager, can_construct)
{
    // Arrange
    // Act
    // Assert
    ASSERT_NO_THROW(ActivityManager activityManager);
}

TEST_F(activity_manager, can_add_digital_pulse) {
    // Arrange
    ActivityManager activityManager;

    // Act
    bool added = activityManager.AddDigitalPulse(1, 2, 3, 4);
    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 1);
    ASSERT_EQ(added, true);
}

TEST_F(activity_manager, can_add_analogue_measure) {
    // Arrange
    ActivityManager activityManager;

    // Act
    bool added = activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 1);
    ASSERT_EQ(added, true);
}

TEST_F(activity_manager, add_fails_when_max_reached) {
    // Arrange
    ActivityManager activityManager;
    bool added = false;
    // Act
    for (uint8_t index = 0; index < MAX_ACTIVITIES; index++)
    {
        added = activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    }
    bool actual = activityManager.AddDigitalPulse(1, 2, 3, 4);

    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 10);
    ASSERT_EQ(added, true);
    ASSERT_EQ(actual, false);
}

TEST_F(activity_manager, number_activities_is_correct) {
    // Arrange,
    ActivityManager activityManager;

    // Act
    activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    activityManager.AddDigitalPulse(1, 2, 3, 4);
    activityManager.AddDigitalPulse(1, 2, 3, 4);
    bool added = activityManager.AddDigitalPulse(1, 2, 3, 4);

    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 6);
    ASSERT_EQ(added, true);
}

TEST_F(activity_manager, activities_are_disabled_on_creation) {
    // Arrange
    ActivityManager activityManager;
    bool added = false;
    // Act
    for (uint8_t index = 0; index < MAX_ACTIVITIES; index++)
    {
        added = activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    }

    PeriodicActivity **activities = activityManager.Activities();
    uint8_t index = 0;
    for (index = 0; index < activityManager.NumberActivities(); index++)
    {
        if (activities[index]->is_enabled())
            continue;
    }

    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 10);
    ASSERT_EQ(added, true);
    ASSERT_EQ(index, activityManager.NumberActivities());
}

TEST_F(activity_manager, activities_are_enabled) {
    // Arrange
    ActivityManager activityManager;
    bool added = false;
    // Act
    for (uint8_t index = 0; index < MAX_ACTIVITIES; index++)
    {
        added = activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    }

    activityManager.enable(true);

    PeriodicActivity **activities = activityManager.Activities();
    uint8_t index = 0;
    for (index = 0; index < activityManager.NumberActivities(); index++)
    {
        if (!activities[index]->is_enabled())
            continue;
    }

    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 10);
    ASSERT_EQ(added, true);
    ASSERT_EQ(index, activityManager.NumberActivities());
}

TEST_F(activity_manager, activities_are_disabled) {
    // Arrange
    ActivityManager activityManager;
    bool added = false;
    // Act
    for (uint8_t index = 0; index < MAX_ACTIVITIES; index++)
    {
        added = activityManager.AddAnalogueMeasure(1, 2, 3, 4);
    }

    activityManager.enable(true);
    activityManager.enable(false);

    PeriodicActivity **activities = activityManager.Activities();
    uint8_t index = 0;
    for (index = 0; index < activityManager.NumberActivities(); index++)
    {
        if (activities[index]->is_enabled())
            continue;
    }

    // Assert
    ASSERT_EQ(activityManager.NumberActivities(), 10);
    ASSERT_EQ(added, true);
    ASSERT_EQ(index, activityManager.NumberActivities());
}