#import "cluster_menu_darwin.h"

static NSMenu *clusterMenu;
static NSMenuItem *clusterStatusItem;
static NSMenuItem *clusterModeItem;
static NSMutableArray<NSMenuItem *> *clusterShareItems;
static NSMenuItem *clusterPlacementItem;
static NSMenuItem *clusterWaterfillItem;
static NSMenuItem *clusterGreedyItem;

// A standard menu item: AppKit draws the checkmark (on), nothing (off) or
// a dash (mixed, used while the server hasn't answered yet).
static NSMenuItem *clusterItem(NSString *title, SEL action, id target) {
    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:action keyEquivalent:@""];
    [item setTarget:target];
    return item;
}

// "On"/"Off", or why the menu can't change it.
static NSString *clusterToolTip(bool on, bool locked, NSString *envVar) {
    if (locked) {
        return [NSString stringWithFormat:@"%@ (set by the %@ environment variable)", on ? @"On" : @"Off", envVar];
    }
    return on ? @"On" : @"Off";
}

@implementation AppDelegate (Cluster)

- (void)addClusterMenuItemsTo:(NSMenu *)menu {
    [menu addItem:[NSMenuItem separatorItem]];
    // Only shown until the server has answered once.
    clusterStatusItem = [[NSMenuItem alloc] initWithTitle:@"Cluster: waiting for the server" action:nil keyEquivalent:@""];
    [clusterStatusItem setEnabled:NO];
    [menu addItem:clusterStatusItem];
    clusterModeItem = clusterItem(@"Cluster Mode", @selector(toggleClusterMode:), self);
    [menu addItem:clusterModeItem];
    clusterShareItems = [[NSMutableArray alloc] init]; // filled in by refreshClusterMenuState

    // OLLAMA_CLUSTER_PLACEMENT: a pick-one submenu, checkmarked like any other.
    NSMenu *placement = [[NSMenu alloc] init];
    clusterWaterfillItem = [placement addItemWithTitle:@"Waterfill"
                                                action:@selector(pickClusterPlacement:)
                                         keyEquivalent:@""];
    [clusterWaterfillItem setTarget:self];
    [clusterWaterfillItem setToolTip:@"Weigh peers by load and latency; can spread a model across several"];
    clusterGreedyItem = [placement addItemWithTitle:@"Greedy"
                                             action:@selector(pickClusterPlacement:)
                                      keyEquivalent:@""];
    [clusterGreedyItem setTarget:self];
    [clusterGreedyItem setToolTip:@"Use the peers with the most free memory, as few as possible"];
    clusterPlacementItem = [[NSMenuItem alloc] initWithTitle:@"Placement" action:nil keyEquivalent:@""];
    [clusterPlacementItem setSubmenu:placement];
    [menu addItem:clusterPlacementItem];
    clusterMenu = menu;

    [self refreshClusterMenuState];
}

// Called on menuWillOpen, so a change made in Settings or with
// `ollama cluster` shows up here too. Draws the last known settings at
// once and asks the server for fresh ones in the background: the request
// can be slow, and a failed one leaves the last settings on screen.
- (void)refreshClusterMenuState {
    [self drawClusterMenuState];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        RefreshClusterMenu();
        dispatch_async(dispatch_get_main_queue(), ^{
            [self drawClusterMenuState];
        });
    });
}

- (void)drawClusterMenuState {
    bool enabled = false, share = false, greedy = false;
    bool enabledLocked = false, shareLocked = false, placementLocked = false;
    // Known once the server has answered; before that, nothing is shown as on or off.
    BOOL reachable = ClusterMenuState(&enabled, &share, &greedy, &enabledLocked, &shareLocked, &placementLocked);

    [clusterStatusItem setHidden:reachable];
    [clusterModeItem setState:!reachable ? NSControlStateValueMixed : enabled ? NSControlStateValueOn : NSControlStateValueOff];
    [clusterModeItem setToolTip:clusterToolTip(enabled, enabledLocked, @"OLLAMA_CLUSTER")];
    [clusterPlacementItem setToolTip:placementLocked ? @"Set by the OLLAMA_CLUSTER_PLACEMENT environment variable" : nil];

    // One switch per GPU, rebuilt each time as GPUs and share_devices can change.
    NSInteger at = [clusterMenu indexOfItem:clusterModeItem] + 1;
    for (NSMenuItem *old in clusterShareItems) {
        [clusterMenu removeItem:old];
    }
    [clusterShareItems removeAllObjects];
    char *lines = ClusterShareItems();
    for (NSString *line in [[NSString stringWithUTF8String:lines] componentsSeparatedByString:@"\n"]) {
        NSArray<NSString *> *f = [line componentsSeparatedByString:@"\t"];
        if (f.count != 3) {
            continue;
        }
        NSMenuItem *item = clusterItem(f[1], @selector(toggleClusterShare:), self);
        [item setRepresentedObject:f[0]];
        BOOL on = [f[2] isEqualToString:@"1"];
        [item setState:!reachable ? NSControlStateValueMixed : on ? NSControlStateValueOn : NSControlStateValueOff];
        [item setToolTip:clusterToolTip(on, shareLocked, @"OLLAMA_CLUSTER_SHARE or OLLAMA_CLUSTER_SHARE_DEVICES")];
        [item setEnabled:reachable && enabled && !shareLocked];
        [clusterMenu insertItem:item atIndex:at++];
        [clusterShareItems addObject:item];
        [item release]; // clusterItem returns it owned; the menu and the array hold it now
    }
    free(lines);
    [clusterWaterfillItem setState:greedy ? NSControlStateValueOff : NSControlStateValueOn];
    [clusterGreedyItem setState:greedy ? NSControlStateValueOn : NSControlStateValueOff];

    // Sharing and placement only mean something while cluster mode is on
    // (same as Settings). Nothing is clickable until the server has answered
    // once, or when an environment variable decides the setting.
    [clusterModeItem setEnabled:reachable && !enabledLocked];
    [clusterPlacementItem setEnabled:reachable && enabled && !placementLocked];
}

- (void)toggleClusterMode:(NSMenuItem *)sender {
    SetClusterModeEnabled([sender state] != NSControlStateValueOn);
    [self drawClusterMenuState];
}

- (void)toggleClusterShare:(NSMenuItem *)sender {
    ToggleClusterShare((char *)[(NSString *)[sender representedObject] UTF8String]);
    [self drawClusterMenuState];
}

- (void)pickClusterPlacement:(NSMenuItem *)sender {
    if ([sender state] != NSControlStateValueOn) {
        SetClusterPlacementGreedy(sender == clusterGreedyItem);
    }
    [self drawClusterMenuState];
}

@end
