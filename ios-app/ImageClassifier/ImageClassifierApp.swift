//
//  ImageClassifierApp.swift
//  ImageClassifier
//
//  Created by Pau Labarta Bajo on 18/9/25.
//

import SwiftUI

@main
struct ImageClassifierApp: App {
    @State private var model = ClassificationModel()
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
        }
    }
}
