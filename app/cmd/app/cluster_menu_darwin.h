#import "app_darwin.h"

// Cluster-mode items in the menu-bar menu, kept out of app_darwin.m so the
// upstream file only carries the hook calls. Go side: cluster_menu_darwin.go.

bool ClusterMenuState(bool *enabled, bool *share, bool *greedy,
                      bool *enabledLocked, bool *shareLocked, bool *placementLocked);
void SetClusterModeEnabled(bool enabled);
void ToggleClusterShare(char *name);
void SetClusterPlacementGreedy(bool greedy);
char *ClusterShareItems(void);

@interface AppDelegate (Cluster)
- (void)addClusterMenuItemsTo:(NSMenu *)menu;
- (void)refreshClusterMenuState;
@end
