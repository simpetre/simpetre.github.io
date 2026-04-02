# Polyfill for Ruby 3.2+ which removed tainted?/untaint
# Required for liquid-4.0.x compatibility
if RUBY_VERSION >= "3.2"
  class Object
    def tainted?
      false
    end

    def untaint
      self
    end
  end
end
